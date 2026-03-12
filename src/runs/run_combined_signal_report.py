import sys
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Ensure `src/` is on PYTHONPATH when running as: `python src/runs/run_combined_signal_report.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from backtest.universe import UniverseSpec, build_liquidity_universe_mask, apply_universe_mask
from backtest.engine import get_rebalance_dates

from runs.run_signal_correlation import compute_mean_rank_corr_matrix
from runs.run_incremental_sharpe import (
    orthogonalize_signals_rank_space,
    compute_daily_rank_ic_roll,
    normalize_weights,
)


def _plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, title: str) -> None:
    """Matplotlib heatmap with numeric annotations."""
    mat = corr.to_numpy(dtype=float)
    labels = list(corr.columns)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(mat, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _weights_summary(Wn: pd.DataFrame) -> pd.DataFrame:
    """Summarize weight time series."""
    out = pd.DataFrame(index=Wn.columns)
    out.index.name = "signal"
    out["mean_w"] = Wn.mean()
    out["std_w"] = Wn.std(ddof=1)
    out["mean_abs_w"] = Wn.abs().mean()
    out["pct_pos"] = (Wn > 0).mean()
    out["pct_neg"] = (Wn < 0).mean()
    out["pct_nan"] = Wn.isna().mean()
    out["max_w"] = Wn.max()
    out["min_w"] = Wn.min()
    return out.sort_values("mean_abs_w", ascending=False)


def _safe_tag(text: str) -> str:
    """Make a filesystem-safe tag."""
    return (
        str(text)
        .replace("/", "_")
        .replace(" ", "")
        .replace(",", "-")
        .replace(":", "-")
        .replace("=", "-")
    )


def _load_incremental_csv(
    horizon: int,
    roll: int,
    out_dir: Path,
    orth: bool,
    orth_threshold: float,
    signals_used: list[str],
    rebalance_frequency: str,
    rebalance_day: str,
) -> tuple[pd.DataFrame, Path]:
    signal_tag = _safe_tag("-".join(signals_used))
    rebalance_tag = _safe_tag(f"{rebalance_frequency}-{rebalance_day}")

    tag = f"__SIG={signal_tag}__RB={rebalance_tag}"
    if orth:
        tag += f"__ORTH_thr{orth_threshold:.2f}"

    path = out_dir / f"incremental_sharpe_ENGINE_roll{roll}_H{horizon}{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing incremental Sharpe output: {path}. Run run_incremental_sharpe.py first."
        )
    return pd.read_csv(path), path


def _performance_lift_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance lift metrics from incremental Sharpe table."""
    df = df.copy()
    if "portfolio" not in df.columns or "sharpe" not in df.columns:
        raise KeyError("Expected columns ['portfolio','sharpe'] in incremental output")

    full = df.loc[df["portfolio"] == "composite_full"].copy()
    if full.empty:
        raise ValueError("incremental output missing 'composite_full'")

    full_sh = float(full["sharpe"].iloc[0])

    base = df[~df["portfolio"].str.startswith("composite_")].copy()
    best_base = float(base["sharpe"].max())
    med_base = float(base["sharpe"].median())
    mean_base = float(base["sharpe"].mean())

    out = pd.DataFrame(
        {
            "metric": [
                "composite_sharpe",
                "best_single_sharpe",
                "median_single_sharpe",
                "mean_single_sharpe",
                "lift_vs_best",
                "lift_vs_median",
                "lift_vs_mean",
            ],
            "value": [
                full_sh,
                best_base,
                med_base,
                mean_base,
                full_sh - best_base,
                full_sh - med_base,
                full_sh - mean_base,
            ],
        }
    )
    return out


def main(
    horizon: int = 5,
    roll: int = 252,
    orth: bool = False,
    orth_threshold: float = 0.30,
    rebalance_frequency: str = "W",
    rebalance_day: str = "FRI",
    data_path: str = "data/processed/crsp_daily_with_lagged_funda_phase3.parquet",
    out_dir_str: str = "data/outputs/multi_signal",
    exclude_signals: str = "",
    signals: str = "",
    liquidity_filter_type: str = "ADV",
    liquidity_top_n: int = 1000,
    liquidity_recon_freq: str = "M",
    adv_window: int = 20,
):
    # -----------------------------
    # 1) Build panel and signals (same base as incremental)
    # -----------------------------
    panel = pd.read_parquet(data_path)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    if ("ret" not in panel.columns) and ("ret_total" in panel.columns):
        panel["ret"] = panel["ret_total"]

    if signals:
        raw_signals = [s.strip() for s in signals.split(",") if s.strip()]
    else:
        raise ValueError(
            "You must provide --signals with columns that already exist in the parquet."
        )

    missing = [s for s in raw_signals if s not in panel.columns]
    if missing:
        raise KeyError(f"Signals not found in panel: {missing}")

    # Optional: exclude signals by name (comma-separated)
    excl = [s.strip() for s in str(exclude_signals).split(",") if s.strip()]
    if excl:
        raw_signals = [s for s in raw_signals if s not in set(excl)]

    # Universe (ADV_1000)
    univ_spec = UniverseSpec(
        filter_type=str(liquidity_filter_type),
        top_n=int(liquidity_top_n),
        recon_freq=str(liquidity_recon_freq),
        adv_window=int(adv_window),
    )
    mask = build_liquidity_universe_mask(panel, univ_spec)
    panel = apply_universe_mask(panel, mask)

    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(rebalance_frequency),
            "REBALANCE_DAY": str(rebalance_day),
        },
    )
    panel = panel[panel["date"].isin(rebalance_dates)].copy()

    # -----------------------------
    # 2) Correlation matrices (rank space)
    # -----------------------------
    corr_pre, _ = compute_mean_rank_corr_matrix(panel, raw_signals)

    orth_meta = None
    signals_used = raw_signals
    corr_post = None

    if orth:
        panel, orth_cols, orth_meta = orthogonalize_signals_rank_space(
            panel,
            signals=raw_signals,
            abs_corr_threshold=float(orth_threshold),
        )
        signals_used = orth_cols
        corr_post, _ = compute_mean_rank_corr_matrix(panel, signals_used)

    # -----------------------------
    # 3) Weights: rolling IC weights series (optimized)
    # Compute forward returns once, then reuse them for all signals.
    # -----------------------------
    df_ic = panel[["permno", "date", "ret_total"] + signals_used].dropna().copy()
    df_ic = df_ic.sort_values(["permno", "date"])

    one_plus = 1.0 + df_ic["ret_total"].astype(float)
    fwd = (
        one_plus.groupby(df_ic["permno"]).shift(-1)
        .groupby(df_ic["permno"]).rolling(horizon, min_periods=horizon)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        .shift(-(horizon - 1))
    )
    df_ic["fwd_ret_h"] = fwd - 1.0
    df_ic = df_ic.dropna(subset=["fwd_ret_h"])

    w_series = {}
    for s in signals_used:
        tmp = df_ic[["date", s, "fwd_ret_h"]].copy()
        tmp["sig_rank"] = tmp.groupby("date")[s].rank(pct=True)
        tmp["ret_rank"] = tmp.groupby("date")["fwd_ret_h"].rank(pct=True)

        ic = (
            tmp.groupby("date")[["sig_rank", "ret_rank"]]
            .apply(lambda x: x["sig_rank"].corr(x["ret_rank"]))
            .rename("rank_ic")
            .sort_index()
        )
        w_series[s] = ic.rolling(roll, min_periods=roll).mean()

    W = pd.DataFrame(w_series).sort_index()
    Wn = normalize_weights(W)

    # -----------------------------
    # 4) Performance lift + marginal contributions (from incremental output)
    # -----------------------------
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    inc, inc_path = _load_incremental_csv(
        horizon=horizon,
        roll=roll,
        out_dir=out_dir,
        orth=orth,
        orth_threshold=orth_threshold,
        signals_used=signals_used,
        rebalance_frequency=rebalance_frequency,
        rebalance_day=rebalance_day,
    )
    perf_lift = _performance_lift_table(inc)

    marg = inc[inc["portfolio"].str.startswith("composite_no_")].copy()
    keep_cols = [
        c
        for c in ["portfolio", "sharpe", "ann_vol", "avg_turnover", "max_drawdown", "delta_sharpe_vs_full"]
        if c in marg.columns
    ]
    marg = marg[keep_cols].sort_values("delta_sharpe_vs_full", ascending=False)

    # -----------------------------
    # 5) Save artifacts
    # -----------------------------
    signal_tag = _safe_tag("-".join(signals_used))
    rebalance_tag = _safe_tag(f"{rebalance_frequency}-{rebalance_day}")

    tag = f"SIG={signal_tag}__RB={rebalance_tag}"
    if orth:
        tag += f"__ORTH_thr{orth_threshold:.2f}"
    else:
        tag += "__RAW"

    # Correlation heatmaps + csv
    corr_pre.to_csv(out_dir / f"combined_corr_pre_{tag}.csv")
    _plot_corr_heatmap(
        corr_pre,
        out_dir / f"combined_corr_pre_{tag}.png",
        title=f"Mean Rank Correlation (pre) [{tag}]",
    )

    if corr_post is not None:
        corr_post.to_csv(out_dir / f"combined_corr_post_{tag}.csv")
        _plot_corr_heatmap(
            corr_post,
            out_dir / f"combined_corr_post_{tag}.png",
            title=f"Mean Rank Correlation (post) [{tag}]",
        )

    # Weights
    Wn.to_csv(out_dir / f"combined_weights_daily_{tag}_roll{roll}_H{horizon}.csv")
    wsum = _weights_summary(Wn)
    wsum.to_csv(out_dir / f"combined_weights_summary_{tag}_roll{roll}_H{horizon}.csv")

    # Performance lift + marginal
    perf_lift.to_csv(out_dir / f"combined_performance_lift_{tag}_roll{roll}_H{horizon}.csv", index=False)
    marg.to_csv(out_dir / f"combined_marginal_{tag}_roll{roll}_H{horizon}.csv", index=False)

    meta = {
        "horizon_for_weights": horizon,
        "roll_for_weights": roll,
        "rebalance_frequency": str(rebalance_frequency),
        "rebalance_day": str(rebalance_day),
        "use_orth": orth,
        "orth_threshold": float(orth_threshold),
        "orth_meta": orth_meta,
        "signals_raw": raw_signals,
        "signals_used": signals_used,
        "incremental_source": str(inc_path),
        "outputs": {
            "corr_pre_png": f"combined_corr_pre_{tag}.png",
            "corr_pre_csv": f"combined_corr_pre_{tag}.csv",
            "corr_post_png": (f"combined_corr_post_{tag}.png" if corr_post is not None else None),
            "corr_post_csv": (f"combined_corr_post_{tag}.csv" if corr_post is not None else None),
            "weights_daily_csv": f"combined_weights_daily_{tag}_roll{roll}_H{horizon}.csv",
            "weights_summary_csv": f"combined_weights_summary_{tag}_roll{roll}_H{horizon}.csv",
            "performance_lift_csv": f"combined_performance_lift_{tag}_roll{roll}_H{horizon}.csv",
            "marginal_csv": f"combined_marginal_{tag}_roll{roll}_H{horizon}.csv",
        },
    }

    with open(out_dir / f"combined_report_meta_{tag}_roll{roll}_H{horizon}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved combined-signal report artifacts to: {out_dir}")
    print(f"Tag: {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 7 combined-signal report: correlation heatmaps, weights, performance lift, marginal contributions."
    )
    parser.add_argument("--horizon", type=int, default=5, help="Horizon used for rolling IC weights.")
    parser.add_argument("--roll", type=int, default=252, help="Rolling window for IC weights.")
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="W",
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
        help="Rebalance frequency used for the report research grid.",
    )
    parser.add_argument(
        "--rebalance-day",
        type=str,
        default="FRI",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        help="Weekday used when rebalance frequency is W/BW/TW.",
    )
    parser.add_argument("--orth", action="store_true", help="Use orthogonalized signals (rank-space residualization).")
    parser.add_argument(
        "--orth_threshold", type=float, default=0.30, help="Absolute rank-correlation threshold for orthogonalization."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/crsp_daily_with_lagged_funda_phase3.parquet",
        help="Path to input daily panel parquet.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/outputs/multi_signal",
        help="Directory to write combined-signal artifacts.",
    )
    parser.add_argument(
        "--exclude_signals",
        type=str,
        default="",
        help="Comma-separated signal names to exclude (e.g., 'idiosyncratic_vol' or 'beta,momentum12-1').",
    )
    parser.add_argument(
        "--signals",
        type=str,
        default="",
        help="Comma-separated signal columns already present in the parquet (e.g. 'op,be,idiosyncratic_vol,beta_sig,momentum12_1').",
    )
    parser.add_argument(
        "--liquidity_filter_type",
        type=str,
        default="ADV",
        help="Liquidity universe filter type (default ADV).",
    )
    parser.add_argument(
        "--liquidity_top_n",
        type=int,
        default=1000,
        help="Universe size for liquidity filter (default 1000).",
    )
    parser.add_argument(
        "--liquidity_recon_freq",
        type=str,
        default="M",
        help="Universe reconstitution frequency (default M).",
    )
    parser.add_argument(
        "--adv_window",
        type=int,
        default=20,
        help="ADV lookback window in trading days (default 20).",
    )

    args = parser.parse_args()
    main(
        horizon=args.horizon,
        roll=args.roll,
        rebalance_frequency=args.rebalance_frequency,
        rebalance_day=args.rebalance_day,
        orth=args.orth,
        orth_threshold=args.orth_threshold,
        data_path=args.data,
        out_dir_str=args.out_dir,
        exclude_signals=args.exclude_signals,
        signals=args.signals,
        liquidity_filter_type=args.liquidity_filter_type,
        liquidity_top_n=args.liquidity_top_n,
        liquidity_recon_freq=args.liquidity_recon_freq,
        adv_window=args.adv_window,
    )