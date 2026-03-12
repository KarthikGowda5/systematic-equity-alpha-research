import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Ensure project root and `src/` are on PYTHONPATH when running as:
# `python src/runs/run_cluster_orthogonalization.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.universe import UniverseSpec, build_liquidity_universe_mask, apply_universe_mask
from backtest.engine import get_rebalance_dates
from runs.run_signal_correlation import compute_mean_rank_corr_matrix, prepare_signals


CLUSTER_NAME_TO_SIGNALS = {
    "value": ["op", "be"],
    "risk": ["idiosyncratic_vol", "beta_sig"],
    "momentum": ["momentum12_1", "Residual_mom", "STR"],
}


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals with intercept using least squares."""
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


def _parse_cluster_specs(cluster_args: list[str]) -> list[tuple[str, list[str]]]:
    """Parse cluster specifications.

    Supported forms:
      --clusters value risk momentum
      --clusters core=op,beta_sig fast=STR,Residual_mom
    """
    cluster_specs: list[tuple[str, list[str]]] = []

    for raw in cluster_args:
        raw = raw.strip()
        if not raw:
            continue

        if "=" in raw:
            name, members = raw.split("=", 1)
            sigs = [s.strip() for s in members.split(",") if s.strip()]
            if not sigs:
                raise ValueError(f"Cluster '{name}' has no signals: {raw}")
            cluster_specs.append((name.strip(), sigs))
        else:
            if raw not in CLUSTER_NAME_TO_SIGNALS:
                raise ValueError(
                    f"Unknown built-in cluster '{raw}'. Available built-ins: {sorted(CLUSTER_NAME_TO_SIGNALS)}"
                )
            cluster_specs.append((raw, list(CLUSTER_NAME_TO_SIGNALS[raw])))

    if not cluster_specs:
        raise ValueError("No valid clusters provided.")

    return cluster_specs


def _subset_corr(mean_corr: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rank_cols = [f"{s}_rank" for s in signals]
    missing_rows = [c for c in rank_cols if c not in mean_corr.index]
    missing_cols = [c for c in rank_cols if c not in mean_corr.columns]
    if missing_rows or missing_cols:
        raise KeyError(
            f"Correlation matrix missing requested cluster signals. Missing rows={missing_rows}, missing cols={missing_cols}"
        )
    out = mean_corr.loc[rank_cols, rank_cols].copy()
    out.index.name = "signal_1"
    out.columns.name = "signal_2"
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Cluster-wise orthogonalization in cross-sectional rank space."
    )
    parser.add_argument(
        "--data",
        default="data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
        help="Input panel parquet. Existing signal columns are used directly; missing ones are computed.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Ordered signal list used for the run. Order matters within each cluster.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        required=True,
        help=(
            "Cluster definitions. Use built-ins like 'value risk momentum' or custom groups like "
            "'core=op,beta_sig fast=STR,Residual_mom'."
        ),
    )
    parser.add_argument(
        "--abs-corr-threshold",
        type=float,
        default=0.30,
        help="Absolute mean rank-correlation threshold for selecting pairs to orthogonalize within each cluster.",
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
        "--corr-csv",
        default="data/outputs/multi_signal/mean_rank_corr_matrix.csv",
        help="Existing pre-orth mean rank-correlation matrix to reuse if compatible.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/outputs/multi_signal",
        help="Directory to save cluster-orth outputs.",
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.data)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    panel, signals = prepare_signals(panel, list(args.signals))

    signal_set = set(signals)
    cluster_specs = _parse_cluster_specs(args.clusters)

    # Validate cluster members are present in signal list and preserve the user-provided signal order.
    ordered_clusters: list[tuple[str, list[str]]] = []
    for cluster_name, cluster_members in cluster_specs:
        missing = [s for s in cluster_members if s not in signal_set]
        if missing:
            raise KeyError(
                f"Cluster '{cluster_name}' includes signals not present in --signals: {missing}"
            )
        ordered = [s for s in signals if s in set(cluster_members)]
        if len(ordered) < 2:
            raise ValueError(
                f"Cluster '{cluster_name}' must contain at least two signals from --signals. Got: {ordered}"
            )
        ordered_clusters.append((cluster_name, ordered))

    # Select rebalance research grid using the same logic as the backtest engine
    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(args.rebalance_frequency),
            "REBALANCE_DAY": str(args.rebalance_day),
        },
    )
    panel = panel[panel["date"].isin(rebalance_dates)].copy()

    univ_spec = UniverseSpec(filter_type="ADV", top_n=1000, recon_freq="M", adv_window=20)
    mask = build_liquidity_universe_mask(panel, univ_spec)
    panel = apply_universe_mask(panel, mask)

    # Build / load full pre-orth correlation matrix for the selected signal set.
    corr_csv = Path(args.corr_csv)
    try:
        mean_corr = pd.read_csv(corr_csv, index_col=0)
        full_rank_cols = [f"{s}_rank" for s in signals]
        mean_corr = mean_corr.loc[full_rank_cols, full_rank_cols].copy()
        mean_corr.index.name = "signal_1"
        mean_corr.columns.name = "signal_2"
        print(f"Using existing correlation matrix: {corr_csv}")
    except Exception as e:
        print(f"Could not reuse existing correlation matrix ({e}); recomputing from panel.")
        mean_corr, _ = compute_mean_rank_corr_matrix(panel, signals)
        mean_corr.index.name = "signal_1"
        mean_corr.columns.name = "signal_2"

    panel = panel.dropna(subset=signals).copy()
    for s in signals:
        panel[f"{s}_rank"] = panel.groupby("date")[s].rank(pct=True)

    cluster_meta: dict[str, dict] = {}
    cluster_post_corrs: dict[str, pd.DataFrame] = {}
    all_orth_cols: list[str] = []

    for cluster_name, cluster_signals in ordered_clusters:
        cluster_corr = _subset_corr(mean_corr, cluster_signals)
        rank_cols = [f"{s}_rank" for s in cluster_signals]
        rank_order = {c: i for i, c in enumerate(rank_cols)}

        pairs = []
        for i, a in enumerate(rank_cols):
            for j, b in enumerate(rank_cols):
                if j <= i:
                    continue
                val = float(cluster_corr.loc[a, b])
                if np.isfinite(val) and abs(val) >= args.abs_corr_threshold:
                    pairs.append((a, b, val))

        controls_for_target: dict[str, list[str]] = {}
        for a, b, _ in pairs:
            if rank_order[a] < rank_order[b]:
                control, target = a, b
            else:
                control, target = b, a
            controls_for_target.setdefault(target, []).append(control)

        cluster_orth_cols: list[str] = []
        for s in cluster_signals:
            orth_col = f"{s}_cluster_orth_rank"
            panel[orth_col] = panel[f"{s}_rank"]
            cluster_orth_cols.append(orth_col)
            all_orth_cols.append(orth_col)

        for target_rank, x_ranks in controls_for_target.items():
            target_sig = target_rank.replace("_rank", "")
            orth_col = f"{target_sig}_cluster_orth_rank"
            panel[orth_col] = residualize_cross_section(
                panel,
                y_col=target_rank,
                x_cols=sorted(set(x_ranks)),
            )

        corr_by_date = panel.groupby("date")[cluster_orth_cols].corr()
        corr_stack = corr_by_date.stack().reset_index()
        corr_stack.columns = ["date", "signal_1", "signal_2", "corr"]
        mean_corr_post = corr_stack.groupby(["signal_1", "signal_2"])["corr"].mean().unstack()
        mean_corr_post.index.name = "signal_1"
        mean_corr_post.columns.name = "signal_2"
        cluster_post_corrs[cluster_name] = mean_corr_post

        cluster_meta[cluster_name] = {
            "signals": cluster_signals,
            "rank_cols": rank_cols,
            "orth_cols": cluster_orth_cols,
            "selected_pairs": [{"a": a, "b": b, "corr": float(v)} for a, b, v in pairs],
            "controls_for_target": controls_for_target,
        }

    print("\n=== Cluster orthogonalization summary ===")
    for cluster_name, meta in cluster_meta.items():
        print(f"\n[{cluster_name}] signals: {meta['signals']}")
        if not meta["selected_pairs"]:
            print("  No pairs exceeded threshold.")
        else:
            for p in meta["selected_pairs"]:
                print(f"  {p['a']} vs {p['b']}: {p['corr']:+.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"cluster_orth_thr{args.abs_corr_threshold:.2f}"
    panel_name = f"panel_with_{tag}.parquet"
    meta_name = f"{tag}.meta.json"

    keep_cols = ["permno", "date"] + signals + [f"{s}_rank" for s in signals] + all_orth_cols
    panel[keep_cols].to_parquet(out_dir / panel_name, index=False)

    # Save per-cluster pre/post correlation matrices.
    for cluster_name, meta in cluster_meta.items():
        pre = _subset_corr(mean_corr, meta["signals"])
        post = cluster_post_corrs[cluster_name]
        pre.to_csv(out_dir / f"mean_rank_corr_matrix__{cluster_name}__pre_{tag}.csv")
        post.to_csv(out_dir / f"mean_rank_corr_matrix__{cluster_name}__post_{tag}.csv")

    meta = {
        "data": str(args.data),
        "corr_csv": str(args.corr_csv),
        "signals": signals,
        "clusters": cluster_meta,
        "abs_corr_threshold": float(args.abs_corr_threshold),
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
            "order_dependence": "Within each cluster, earlier signals in --signals act as controls for later correlated signals.",
            "scope": "Orthogonalization is only applied within clusters, not across clusters.",
            "panel_output": panel_name,
        },
    }

    with open(out_dir / meta_name, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()