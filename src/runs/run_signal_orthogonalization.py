import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Ensure project root and `src/` are on PYTHONPATH when running as:
# `python src/runs/run_signal_orthogonalization.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root (flagship/)
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.universe import UniverseSpec, build_liquidity_universe_mask, apply_universe_mask
from backtest.engine import get_rebalance_dates
from runs.run_signal_correlation import compute_mean_rank_corr_matrix, prepare_signals


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals with intercept using least squares. y: (n,), X: (n,k)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    return y - y_hat


def residualize_cross_section(panel: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.Series:
    """Per-date cross-sectional residualization of y_col on x_cols."""
    out = pd.Series(index=panel.index, dtype=float)

    for _, sub in panel.groupby("date", sort=False):
        sub2 = sub[[y_col] + x_cols].dropna()
        if sub2.shape[0] < (len(x_cols) + 5):
            continue

        y = sub2[y_col].to_numpy(dtype=float)
        X = sub2[x_cols].to_numpy(dtype=float)
        resid = _ols_residuals(y, X)
        out.loc[sub2.index] = resid

    return out


def _load_existing_rank_corr_matrix(corr_csv: Path, signals: list[str]) -> pd.DataFrame:
    """Load an existing rank-correlation matrix and subset/reorder it to the requested signals.

    The stored matrix is expected to use rank-column names like '<signal>_rank'.
    """
    if not corr_csv.exists():
        raise FileNotFoundError(f"Correlation matrix not found: {corr_csv}")

    mean_corr = pd.read_csv(corr_csv, index_col=0)
    rank_cols = [f"{s}_rank" for s in signals]

    missing_rows = [c for c in rank_cols if c not in mean_corr.index]
    missing_cols = [c for c in rank_cols if c not in mean_corr.columns]
    if missing_rows or missing_cols:
        raise KeyError(
            f"Existing correlation matrix does not contain all requested signals. "
            f"Missing rows={missing_rows}, missing cols={missing_cols}. Available rows={list(mean_corr.index)}"
        )

    mean_corr = mean_corr.loc[rank_cols, rank_cols].copy()
    mean_corr.index.name = "signal_1"
    mean_corr.columns.name = "signal_2"
    return mean_corr


def main():
    parser = argparse.ArgumentParser(description="Orthogonalize selected signals in cross-sectional rank space.")
    parser.add_argument(
        "--data",
        default="data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
        help="Input panel parquet. Existing signal columns are used directly; missing ones are computed.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Signal columns to use. Order matters: earlier signals are treated as controls for later correlated signals.",
    )
    parser.add_argument(
        "--abs-corr-threshold",
        type=float,
        default=0.30,
        help="Absolute mean rank-correlation threshold for selecting pairs to orthogonalize.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        default="W",
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
        help="Rebalance frequency used to select the research grid.",
    )
    parser.add_argument(
        "--rebalance-day",
        default="FRI",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        help="Weekday used when rebalance frequency is W/BW/TW.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/outputs/multi_signal",
        help="Directory to save orthogonalization outputs.",
    )
    parser.add_argument(
        "--corr-csv",
        default="data/outputs/multi_signal/mean_rank_corr_matrix.csv",
        help="Existing pre-orth mean rank-correlation matrix to reuse. If missing or incompatible, the script recomputes it.",
    )
    args = parser.parse_args()

    # 1) Load panel
    panel = pd.read_parquet(args.data)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    # 2) Ensure requested signals exist in the panel, computing only missing ones.
    panel, signals = prepare_signals(panel, list(args.signals))

    # 3) Rebalance-date research grid
    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(args.rebalance_frequency),
            "REBALANCE_DAY": str(args.rebalance_day),
        },
    )
    panel = panel[panel["date"].isin(rebalance_dates)].copy()

    # 4) Universe: ADV top 1000 using backtest engine logic
    univ_spec = UniverseSpec(filter_type="ADV", top_n=1000, recon_freq="M", adv_window=20)
    mask = build_liquidity_universe_mask(panel, univ_spec)
    panel = apply_universe_mask(panel, mask)

    # 5) Load an existing mean rank correlation matrix if available; otherwise compute it.
    corr_csv = Path(args.corr_csv)
    try:
        mean_corr = _load_existing_rank_corr_matrix(corr_csv, signals)
        rank_cols = [f"{s}_rank" for s in signals]
        print(f"Using existing correlation matrix: {corr_csv}")
    except Exception as e:
        print(f"Could not reuse existing correlation matrix ({e}); recomputing from panel.")
        mean_corr, rank_cols = compute_mean_rank_corr_matrix(panel, signals)
        mean_corr.index.name = "signal_1"
        mean_corr.columns.name = "signal_2"

    # NOTE: compute_mean_rank_corr_matrix builds ranks on an internal copy.
    # For orthogonalization we need the rank columns on `panel` itself.
    panel = panel.dropna(subset=signals).copy()
    for s in signals:
        panel[s + "_rank"] = panel.groupby("date")[s].rank(pct=True)
    rank_cols = [s + "_rank" for s in signals]

    # 6) Select highly correlated pairs.
    # IMPORTANT: signal order is defined by the user-provided `--signals` list.
    # The earlier signal is treated as the control, and the later signal is residualized.
    rank_order = {c: i for i, c in enumerate(rank_cols)}

    pairs = []
    for i, a in enumerate(rank_cols):
        for j, b in enumerate(rank_cols):
            if j <= i:
                continue
            val = float(mean_corr.loc[a, b])
            if np.isfinite(val) and (abs(val) >= args.abs_corr_threshold):
                pairs.append((a, b, val))

    controls_for_target: dict[str, list[str]] = {}
    for a, b, _ in pairs:
        if rank_order[a] < rank_order[b]:
            control, target = a, b
        else:
            control, target = b, a
        controls_for_target.setdefault(target, []).append(control)

    # 7) Orthogonalize selected targets in rank space
    orth_cols: list[str] = []
    for y_col, x_cols in controls_for_target.items():
        orth_col = y_col.replace("_rank", "") + "_orth_rank"
        panel[orth_col] = residualize_cross_section(panel, y_col=y_col, x_cols=sorted(set(x_cols)))
        orth_cols.append(orth_col)

    # Carry forward non-residualized signals as their original ranks
    for rc in rank_cols:
        base = rc.replace("_rank", "")
        orth_col = base + "_orth_rank"
        if orth_col in panel.columns:
            continue
        panel[orth_col] = panel[rc]
        orth_cols.append(orth_col)

    # 8) Correlation matrix after orthogonalization
    corr_by_date = panel.groupby("date")[orth_cols].corr()
    corr_stack = corr_by_date.stack().reset_index()
    corr_stack.columns = ["date", "signal_1", "signal_2", "corr"]
    mean_corr_orth = corr_stack.groupby(["signal_1", "signal_2"])["corr"].mean().unstack()
    mean_corr_orth.index.name = "signal_1"
    mean_corr_orth.columns.name = "signal_2"

    print("\n=== Mean rank correlation (pre-orthogonalization) ===")
    print(mean_corr)

    print(f"\nSelected pairs |corr| >= {args.abs_corr_threshold:.2f} (rank-space):")
    if not pairs:
        print("  (none)")
    else:
        for a, b, val in sorted(pairs, key=lambda x: -abs(x[2])):
            print(f"  {a} vs {b}: {val:+.4f}")

    print("\n=== Mean rank correlation (post-orthogonalization) ===")
    print(mean_corr_orth)

    # 9) Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_name = "mean_rank_corr_matrix_selected.csv"
    post_name = f"mean_rank_corr_matrix_orth_thr{args.abs_corr_threshold:.2f}.csv"
    panel_name = f"panel_with_orth_signals_thr{args.abs_corr_threshold:.2f}.parquet"
    meta_name = f"mean_rank_corr_matrix_orth_thr{args.abs_corr_threshold:.2f}.meta.json"

    mean_corr.to_csv(out_dir / pre_name)
    mean_corr_orth.to_csv(out_dir / post_name)

    keep_cols = ["permno", "date"] + signals + rank_cols + orth_cols
    panel[keep_cols].to_parquet(out_dir / panel_name, index=False)

    meta = {
        "data": str(args.data),
        "corr_csv": str(args.corr_csv),
        "signals": signals,
        "signal_order_used_for_orthogonalization": signals,
        "rank_cols": rank_cols,
        "orth_cols": orth_cols,
        "abs_corr_threshold": float(args.abs_corr_threshold),
        "selected_pairs": [{"a": a, "b": b, "corr": float(v)} for a, b, v in pairs],
        "controls_for_target": controls_for_target,
        "rebalance_frequency": str(args.rebalance_frequency),
        "rebalance_day": args.rebalance_day,
        "universe": {
            "type": "ADV",
            "top_n": 1000,
            "recon_freq": "M",
            "adv_window": 20,
            "source": "backtest.universe",
        },
        "notes": {
            "orth_space": "cross-sectional rank space",
            "order_dependence": "earlier signals in --signals act as controls for later correlated signals",
            "panel_output": panel_name,
        },
    }

    with open(out_dir / meta_name, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()