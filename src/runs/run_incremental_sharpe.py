import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Ensure project root and `src/` are on PYTHONPATH when running as:
# `python src/runs/run_incremental_sharpe.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.universe import UniverseSpec, build_liquidity_universe_mask
from backtest.engine import run_baseline_backtest, get_rebalance_dates
from runs.run_signal_correlation import prepare_signals
from runs.run_cluster_orthogonalization import _parse_cluster_specs


# -----------------------------
# Orthogonalization helpers
# -----------------------------

def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals with intercept using least squares. y: (n,), X: (n,k)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - (X1 @ beta)


def _residualize_cross_section(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.Series:
    """Per-date cross-sectional residualization of y_col on x_cols."""
    out = pd.Series(index=df.index, dtype=float)
    for _, sub in df.groupby("date", sort=False):
        sub2 = sub[[y_col] + x_cols].dropna()
        if sub2.shape[0] < (len(x_cols) + 5):
            continue
        y = sub2[y_col].to_numpy(dtype=float)
        X = sub2[x_cols].to_numpy(dtype=float)
        out.loc[sub2.index] = _ols_residuals(y, X)
    return out


def _mean_rank_corr_matrix_from_panel(panel: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Mean same-date cross-sectional correlation matrix for rank columns built from signals."""
    df = panel.dropna(subset=signals).copy()
    rank_cols = []
    for s in signals:
        rcol = f"{s}_rank"
        df[rcol] = df.groupby("date")[s].rank(pct=True)
        rank_cols.append(rcol)

    per_date_corrs = []
    for _, g in df.groupby("date"):
        if len(g) < 3:
            continue
        c = g[rank_cols].corr(method="pearson")
        per_date_corrs.append(c)

    if not per_date_corrs:
        mean_corr = pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)
    else:
        mean_corr = pd.concat(per_date_corrs).groupby(level=0).mean()
    mean_corr.index.name = "signal_1"
    mean_corr.columns.name = "signal_2"
    return mean_corr, rank_cols


def orthogonalize_signals_rank_space(
    panel: pd.DataFrame,
    signals: list[str],
    abs_corr_threshold: float = 0.30,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Orthogonalize signals using regression residualization in cross-sectional rank space.

    Signal order matters: the later-ordered signal is residualized on the earlier-ordered correlated signal(s).

    Returns:
    - panel with new orth columns added (suffix `_orth`),
    - list of orth signal column names,
    - meta dict with selected pairs and control mapping.

    NOTE: The resulting orth columns are not re-ranked here; the engine will rank them at rebalance.
    """
    df = panel.dropna(subset=signals).copy()

    for s in signals:
        df[s + "_rank"] = df.groupby("date")[s].rank(pct=True)

    mean_corr, rank_cols = _mean_rank_corr_matrix_from_panel(df, signals)
    rank_order = {c: i for i, c in enumerate(rank_cols)}

    pairs = []
    for i, a in enumerate(rank_cols):
        for j, b in enumerate(rank_cols):
            if j <= i:
                continue
            val = float(mean_corr.loc[a, b])
            if np.isfinite(val) and (abs(val) >= abs_corr_threshold):
                pairs.append((a, b, val))

    controls_for_target: dict[str, list[str]] = {}
    for a, b, _ in pairs:
        if rank_order[a] < rank_order[b]:
            control, target = a, b
        else:
            control, target = b, a
        controls_for_target.setdefault(target, []).append(control)

    orth_cols: list[str] = []
    for s in signals:
        orth = s + "_orth"
        df[orth] = df[s + "_rank"]
        orth_cols.append(orth)

    for target_rank, x_ranks in controls_for_target.items():
        target_sig = target_rank.replace("_rank", "")
        orth = target_sig + "_orth"
        df[orth] = _residualize_cross_section(df, y_col=target_rank, x_cols=sorted(set(x_ranks)))

    meta = {
        "abs_corr_threshold": float(abs_corr_threshold),
        "selected_pairs": [{"a": a, "b": b, "corr": float(v)} for a, b, v in pairs],
        "controls_for_target": controls_for_target,
        "signal_order_used_for_orthogonalization": list(signals),
    }

    out = panel.copy()
    out = out.merge(df[["permno", "date"] + orth_cols], on=["permno", "date"], how="left")

    return out, orth_cols, meta


def cluster_orthogonalize_signals_rank_space(
    panel: pd.DataFrame,
    signals: list[str],
    cluster_args: list[str],
    abs_corr_threshold: float = 0.30,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Orthogonalize signals only within user-specified clusters in cross-sectional rank space.

    Signal order matters within each cluster: the later-ordered signal is residualized on
    the earlier-ordered correlated signal(s) inside that cluster.

    Returns:
    - panel with new cluster-orth columns added (suffix `_cluster_orth`),
    - list of cluster-orth signal column names aligned to the input `signals`,
    - meta dict with selected pairs and control mapping per cluster.
    """
    df = panel.dropna(subset=signals).copy()

    for s in signals:
        df[f"{s}_rank"] = df.groupby("date")[s].rank(pct=True)

    cluster_specs = _parse_cluster_specs(cluster_args)
    signal_set = set(signals)

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

    cluster_meta: dict[str, dict] = {}
    assigned_signals: set[str] = set()

    for cluster_name, cluster_signals in ordered_clusters:
        mean_corr, rank_cols = _mean_rank_corr_matrix_from_panel(df, cluster_signals)
        rank_order = {c: i for i, c in enumerate(rank_cols)}

        pairs = []
        if not mean_corr.empty:
            for i, a in enumerate(rank_cols):
                for j, b in enumerate(rank_cols):
                    if j <= i:
                        continue
                    val = float(mean_corr.loc[a, b])
                    if np.isfinite(val) and (abs(val) >= abs_corr_threshold):
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
            orth = f"{s}_cluster_orth"
            df[orth] = df[f"{s}_rank"]
            cluster_orth_cols.append(orth)
            assigned_signals.add(s)

        for target_rank, x_ranks in controls_for_target.items():
            target_sig = target_rank.replace("_rank", "")
            orth = f"{target_sig}_cluster_orth"
            df[orth] = _residualize_cross_section(df, y_col=target_rank, x_cols=sorted(set(x_ranks)))

        cluster_meta[cluster_name] = {
            "signals": cluster_signals,
            "rank_cols": rank_cols,
            "orth_cols": cluster_orth_cols,
            "selected_pairs": [{"a": a, "b": b, "corr": float(v)} for a, b, v in pairs],
            "controls_for_target": controls_for_target,
        }

    # Signals not assigned to any cluster keep their raw rank with the cluster-orth suffix.
    for s in signals:
        if s in assigned_signals:
            continue
        df[f"{s}_cluster_orth"] = df[f"{s}_rank"]

    cluster_orth_cols = [f"{s}_cluster_orth" for s in signals]
    meta = {
        "abs_corr_threshold": float(abs_corr_threshold),
        "clusters": cluster_meta,
        "signal_order_used_for_orthogonalization": list(signals),
    }

    out = panel.copy()
    out = out.merge(df[["permno", "date"] + cluster_orth_cols], on=["permno", "date"], how="left")

    return out, cluster_orth_cols, meta


def compute_forward_total_return(panel: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward horizon-day total return per (permno,date): prod_{k=1..h}(1+ret_{t+k})-1."""
    if "ret_total" not in panel.columns:
        raise KeyError("Expected 'ret_total' in panel to compute forward returns.")

    df = panel[["permno", "date", "ret_total"]].copy()
    df = df.sort_values(["permno", "date"])

    one_plus = 1.0 + df["ret_total"].astype(float)

    fwd_prod = (
        one_plus.groupby(df["permno"]).shift(-1)
        .groupby(df["permno"]).rolling(horizon, min_periods=horizon)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        .shift(-(horizon - 1))
    )

    return fwd_prod - 1.0


def compute_daily_rank_ic_roll(panel: pd.DataFrame, signal_col: str, horizon: int = 5, roll: int = 252) -> pd.Series:
    """Daily cross-sectional rank-IC rolling mean for a signal vs forward horizon-day return."""
    df = panel[["permno", "date", signal_col, "ret_total"]].dropna().copy()
    df = df.sort_values(["permno", "date"])

    df["fwd_ret_h"] = compute_forward_total_return(df, horizon=horizon).values
    df = df.dropna(subset=["fwd_ret_h"]).copy()

    df["sig_rank"] = df.groupby("date")[signal_col].rank(pct=True)
    df["ret_rank"] = df.groupby("date")["fwd_ret_h"].rank(pct=True)

    ic = (
        df.groupby("date")[["sig_rank", "ret_rank"]]
        .apply(lambda x: x["sig_rank"].corr(x["ret_rank"]))
        .rename("rank_ic")
        .sort_index()
    )

    ic_roll = ic.rolling(roll, min_periods=roll).mean()
    ic_roll.name = "ic_roll"
    return ic_roll


def normalize_weights(W: pd.DataFrame) -> pd.DataFrame:
    """Normalize weights by sum(|w|) each date."""
    denom = W.abs().sum(axis=1)
    return W.div(denom.replace(0.0, np.nan), axis=0)


# ----------------------------------------------------------------------
# Helper for making filesystem-safe tags
# ----------------------------------------------------------------------

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


def build_composite_signal(
    panel: pd.DataFrame,
    signals: list[str],
    Wn: pd.DataFrame,
    out_col: str,
) -> pd.DataFrame:
    """Build an IC-weighted composite signal in rank space for ALL dates."""
    df = panel.copy()

    for s in signals:
        df[s + "_rank"] = df.groupby("date")[s].rank(pct=True)

    dates = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    Wn_aligned = Wn.reindex(dates).ffill()

    df[out_col] = 0.0
    for s in signals:
        w_ser = Wn_aligned[s]
        df[out_col] += df[s + "_rank"] * df["date"].map(w_ser)

    return df


def backtest_one(df: pd.DataFrame, signal_col: str, universe_mask: pd.DataFrame, settings: dict) -> dict:
    """Run baseline engine backtest and return a metrics row."""
    res = run_baseline_backtest(
        df=df,
        settings=settings,
        signal_col=signal_col,
        universe_mask=universe_mask,
        id_col="permno",
        date_col="date",
        ret_col="ret_total",
    )

    m = dict(res.metrics)
    m["ann_ret"] = float(m.get("mean_period_return", 0.0) * m.get("periods_per_year", 0.0))
    m["periods"] = int(m.get("periods", 0.0))
    return m


def main(
    horizon: int = 5,
    roll: int = 252,
    signals: list[str] | None = None,
    long_q: float = 0.9,
    short_q: float = 0.1,
    rebalance_frequency: str = "W",
    rebalance_day: str = "FRI",
    use_orth: bool = False,
    use_cluster_orth: bool = False,
    cluster_defs: list[str] | None = None,
    orth_threshold: float = 0.30,
    data_path: str = "data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
    out_dir_str: str = "data/outputs/multi_signal",
):
    if not signals:
        raise ValueError("You must provide at least one signal via --signals.")
    if use_orth and use_cluster_orth:
        raise ValueError("Use only one orthogonalization mode: --orth or --cluster-orth, not both.")
    if use_cluster_orth and not cluster_defs:
        raise ValueError("--cluster-orth requires --clusters.")

    panel = pd.read_parquet(data_path)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    panel, signals = prepare_signals(panel, list(signals))

    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(rebalance_frequency),
            "REBALANCE_DAY": str(rebalance_day),
        },
    )

    univ_spec = UniverseSpec(filter_type="ADV", top_n=1000, recon_freq="M", adv_window=20)
    universe_mask = build_liquidity_universe_mask(panel, univ_spec)

    orth_meta = None
    if use_orth:
        panel, orth_cols, orth_meta = orthogonalize_signals_rank_space(
            panel,
            signals=signals,
            abs_corr_threshold=float(orth_threshold),
        )
        signals = orth_cols
    elif use_cluster_orth:
        panel, orth_cols, orth_meta = cluster_orthogonalize_signals_rank_space(
            panel,
            signals=signals,
            cluster_args=list(cluster_defs or []),
            abs_corr_threshold=float(orth_threshold),
        )
        signals = orth_cols

    w_series = {}
    for s in signals:
        w_series[s] = compute_daily_rank_ic_roll(panel, signal_col=s, horizon=horizon, roll=roll)

    W = pd.DataFrame(w_series).sort_index()
    Wn_full = normalize_weights(W)

    df_full = build_composite_signal(panel, signals, Wn_full, out_col="composite_full")

    for drop in signals:
        keep = [s for s in signals if s != drop]
        Wn_k = normalize_weights(W[keep])
        df_full = build_composite_signal(df_full, keep, Wn_k, out_col=f"composite_no_{drop}")

    settings = {
        "REBALANCE_FREQUENCY": str(rebalance_frequency),
        "REBALANCE_DAY": str(rebalance_day),
        "LONG_QUANTILE": float(long_q),
        "SHORT_QUANTILE": float(short_q),
    }

    rows = []

    for s in signals:
        m = backtest_one(df_full, signal_col=s, universe_mask=universe_mask, settings=settings)
        rows.append({"portfolio": s, **m})

    m = backtest_one(df_full, signal_col="composite_full", universe_mask=universe_mask, settings=settings)
    rows.append({"portfolio": "composite_full", **m})

    for drop in signals:
        col = f"composite_no_{drop}"
        m = backtest_one(df_full, signal_col=col, universe_mask=universe_mask, settings=settings)
        rows.append({"portfolio": col, **m})

    out = pd.DataFrame(rows)

    full_sh = float(out.loc[out["portfolio"] == "composite_full", "sharpe"].iloc[0])
    out["delta_sharpe_vs_full"] = np.nan
    mask_drop = out["portfolio"].str.startswith("composite_no_")
    out.loc[mask_drop, "delta_sharpe_vs_full"] = full_sh - out.loc[mask_drop, "sharpe"].astype(float)

    cols_show = [
        "portfolio",
        "periods",
        "periods_per_year",
        "ann_ret",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "delta_sharpe_vs_full",
    ]
    cols_show = [c for c in cols_show if c in out.columns]

    with pd.option_context("display.max_rows", 200, "display.max_columns", 100, "display.width", 200):
        print(out[cols_show].sort_values("portfolio"))

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_tag = _safe_tag("-".join(signals))
    rebalance_tag = _safe_tag(f"{rebalance_frequency}-{rebalance_day}")

    tag = f"__SIG={signal_tag}__RB={rebalance_tag}"
    if use_orth:
        tag += f"__ORTH_thr{orth_threshold:.2f}"
    elif use_cluster_orth:
        cluster_tag = _safe_tag("-".join(cluster_defs or []))
        tag += f"__CLUSTER_ORTH_thr{orth_threshold:.2f}__CL={cluster_tag}"

    out_path = out_dir / f"incremental_sharpe_ENGINE_roll{roll}_H{horizon}{tag}.csv"
    out.to_csv(out_path, index=False)

    meta = {
        "data": str(data_path),
        "horizon_for_weights": horizon,
        "roll_for_weights": roll,
        "rebalance_frequency": str(rebalance_frequency),
        "rebalance_day": str(rebalance_day),
        "long_quantile": long_q,
        "short_quantile": short_q,
        "use_orth": use_orth,
        "use_cluster_orth": use_cluster_orth,
        "cluster_defs": list(cluster_defs or []),
        "orth_threshold": float(orth_threshold),
        "orth_meta": orth_meta,
        "engine": "backtest.engine.run_baseline_backtest",
        "portfolio_construction": "backtest.portfolio_construction (rank_to_long_short + equal LS weights)",
        "returns": "engine uses compounded daily ret_total between rebalance dates",
        "turnover": "backtest.turnover.simple_turnover",
        "metrics": "backtest.metrics.summarize_performance",
        "signals": signals,
        "composites": ["composite_full"] + [f"composite_no_{s}" for s in signals],
        "universe": {
            "type": "ADV",
            "top_n": 1000,
            "recon_freq": "M",
            "adv_window": 20,
            "source": "backtest.universe.build_liquidity_universe_mask",
        },
    }
    with open(out_dir / f"incremental_sharpe_ENGINE_roll{roll}_H{horizon}{tag}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved incremental Sharpe table to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 incremental Sharpe: base vs IC-weighted composite + drop-one (engine-consistent).")
    parser.add_argument("--horizon", type=int, default=5, help="Horizon used for rolling IC weights (forward return horizon).")
    parser.add_argument("--roll", type=int, default=252, help="Rolling window length (trading days) for IC weights.")
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Signal columns to use. Existing panel columns are used directly; missing ones are computed.",
    )
    parser.add_argument("--long_q", type=float, default=0.9, help="Long quantile.")
    parser.add_argument("--short_q", type=float, default=0.1, help="Short quantile.")
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="W",
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
        help="Rebalance frequency used both for the research grid and the engine backtest.",
    )
    parser.add_argument(
        "--rebalance-day",
        type=str,
        default="FRI",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        help="Weekday used when rebalance frequency is W/BW/TW.",
    )
    parser.add_argument("--orth", action="store_true", help="Use orthogonalized signals (rank-space residualization) before IC weighting/backtest.")
    parser.add_argument(
        "--cluster-orth",
        action="store_true",
        help="Use cluster-wise orthogonalized signals before IC weighting/backtest.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        help=(
            "Cluster definitions for --cluster-orth. Use built-ins like 'value risk momentum' or custom groups like "
            "'core=op,beta_sig fast=STR,Residual_mom'."
        ),
    )
    parser.add_argument("--orth_threshold", type=float, default=0.30, help="Absolute rank-correlation threshold for selecting pairs to orthogonalize.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
        help="Input daily panel parquet path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/outputs/multi_signal",
        help="Output directory for incremental Sharpe artifacts.",
    )

    args = parser.parse_args()
    main(
        horizon=args.horizon,
        roll=args.roll,
        signals=args.signals,
        long_q=args.long_q,
        short_q=args.short_q,
        rebalance_frequency=args.rebalance_frequency,
        rebalance_day=args.rebalance_day,
        use_orth=args.orth,
        use_cluster_orth=args.cluster_orth,
        cluster_defs=args.clusters,
        orth_threshold=args.orth_threshold,
        data_path=args.data,
        out_dir_str=args.out_dir,
    )