import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Ensure project root and `src/` are on PYTHONPATH when running as:
# `python src/runs/run_IC_weightsavg.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root (flagship/)
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.universe import UniverseSpec, build_liquidity_universe_mask, apply_universe_mask
from backtest.engine import get_rebalance_dates
from runs.run_signal_correlation import prepare_signals
from runs.run_cluster_orthogonalization import _parse_cluster_specs


# --- Local helper for orthogonalization (self-contained, no external dependency) ---

def _select_correlated_pairs(mean_corr: pd.DataFrame, abs_corr_threshold: float) -> list[tuple[str, str, float]]:
    """Return list of (a,b,corr) for |corr| >= threshold, using upper triangle without diagonal."""
    pairs: list[tuple[str, str, float]] = []
    cols = list(mean_corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = float(mean_corr.loc[a, b])
            if np.isfinite(c) and abs(c) >= abs_corr_threshold:
                pairs.append((a, b, c))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs


def _residualize_one_date(sub: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.Series:
    """OLS residuals of y on X (with intercept) for a single date cross-section."""
    sub2 = sub[[y_col] + x_cols].dropna()
    if sub2.empty:
        return pd.Series(index=sub.index, dtype=float)

    y = sub2[y_col].astype(float).values
    X = sub2[x_cols].astype(float).values
    X = np.column_stack([np.ones(len(X)), X])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    resid = y - y_hat

    out = pd.Series(index=sub.index, dtype=float)
    out.loc[sub2.index] = resid
    return out


def _mean_rank_corr_matrix_from_panel(panel: pd.DataFrame, rank_cols: list[str]) -> pd.DataFrame:
    """Mean same-date cross-sectional correlation matrix for rank columns."""
    per_date_corrs = []
    for _, g in panel.dropna(subset=rank_cols).groupby("date"):
        if len(g) < 3:
            continue
        c = g[rank_cols].corr(method="pearson")
        per_date_corrs.append(c)

    if not per_date_corrs:
        return pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)

    mean_corr = pd.concat(per_date_corrs).groupby(level=0).mean()
    mean_corr.index.name = "signal_1"
    mean_corr.columns.name = "signal_2"
    return mean_corr


def orthogonalize_signals_rank_space(
    panel: pd.DataFrame,
    signals: list[str],
    abs_corr_threshold: float = 0.20,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Orthogonalize signals in rank space using regression residualization.

    Signal order matters: for a correlated pair, the later signal is residualized on the earlier signal.
    """
    df = panel.copy()

    rank_cols = []
    for s in signals:
        rcol = f"{s}_rank"
        df[rcol] = df.groupby("date")[s].rank(pct=True)
        rank_cols.append(rcol)

    mean_corr = _mean_rank_corr_matrix_from_panel(df, rank_cols)

    if mean_corr.empty:
        orth_cols = [f"{s}_orth_rank" for s in signals]
        for s in signals:
            df[f"{s}_orth_rank"] = df[f"{s}_rank"]
        meta = {
            "abs_corr_threshold": float(abs_corr_threshold),
            "selected_pairs": [],
            "controls_for_target": {},
            "signal_order_used_for_orthogonalization": list(signals),
        }
        return df, orth_cols, meta

    pairs = _select_correlated_pairs(mean_corr, abs_corr_threshold=float(abs_corr_threshold))

    controls_for_target: dict[str, list[str]] = {}
    for a, b, _ in pairs:
        controls_for_target.setdefault(b, []).append(a)

    for s in signals:
        df[f"{s}_orth_rank"] = df[f"{s}_rank"]

    for target_rank, x_cols in controls_for_target.items():
        target_sig = target_rank.replace("_rank", "")
        orth_col = f"{target_sig}_orth_rank"
        x_orth_cols = [c.replace("_rank", "") + "_orth_rank" for c in x_cols]

        resid = df.groupby("date", group_keys=False).apply(
            lambda g: _residualize_one_date(g, y_col=orth_col, x_cols=x_orth_cols)
        )
        df[orth_col] = resid.groupby(df["date"]).rank(pct=True)

    orth_cols = [f"{s}_orth_rank" for s in signals]
    meta = {
        "abs_corr_threshold": float(abs_corr_threshold),
        "selected_pairs": [{"a": a, "b": b, "corr": float(c)} for (a, b, c) in pairs],
        "controls_for_target": controls_for_target,
        "signal_order_used_for_orthogonalization": list(signals),
    }
    return df, orth_cols, meta


def cluster_orthogonalize_signals_rank_space(
    panel: pd.DataFrame,
    signals: list[str],
    cluster_args: list[str],
    abs_corr_threshold: float = 0.20,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Orthogonalize signals only within user-specified clusters in rank space.

    The user-provided `signals` order is preserved within each cluster, and earlier signals
    act as controls for later correlated signals inside that cluster.
    """
    df = panel.copy()

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

    all_cluster_orth_cols: list[str] = []
    cluster_meta: dict[str, dict] = {}

    for cluster_name, cluster_signals in ordered_clusters:
        rank_cols = [f"{s}_rank" for s in cluster_signals]
        mean_corr = _mean_rank_corr_matrix_from_panel(df, rank_cols)

        if mean_corr.empty:
            pairs = []
            controls_for_target: dict[str, list[str]] = {}
        else:
            pairs = _select_correlated_pairs(mean_corr, abs_corr_threshold=float(abs_corr_threshold))
            controls_for_target = {}
            for a, b, _ in pairs:
                controls_for_target.setdefault(b, []).append(a)

        cluster_orth_cols: list[str] = []
        for s in cluster_signals:
            orth_col = f"{s}_cluster_orth_rank"
            df[orth_col] = df[f"{s}_rank"]
            cluster_orth_cols.append(orth_col)
            all_cluster_orth_cols.append(orth_col)

        for target_rank, x_cols in controls_for_target.items():
            target_sig = target_rank.replace("_rank", "")
            orth_col = f"{target_sig}_cluster_orth_rank"
            x_orth_cols = [c.replace("_rank", "") + "_cluster_orth_rank" for c in x_cols]

            resid = df.groupby("date", group_keys=False).apply(
                lambda g: _residualize_one_date(g, y_col=orth_col, x_cols=x_orth_cols)
            )
            df[orth_col] = resid.groupby(df["date"]).rank(pct=True)

        cluster_meta[cluster_name] = {
            "signals": cluster_signals,
            "rank_cols": rank_cols,
            "orth_cols": cluster_orth_cols,
            "selected_pairs": [{"a": a, "b": b, "corr": float(c)} for (a, b, c) in pairs],
            "controls_for_target": controls_for_target,
        }

    # For signals that are not part of any cluster, carry forward raw ranks with the cluster suffix.
    assigned = {c.replace("_cluster_orth_rank", "") for c in all_cluster_orth_cols}
    for s in signals:
        if s in assigned:
            continue
        orth_col = f"{s}_cluster_orth_rank"
        df[orth_col] = df[f"{s}_rank"]
        all_cluster_orth_cols.append(orth_col)

    meta = {
        "abs_corr_threshold": float(abs_corr_threshold),
        "clusters": cluster_meta,
        "signal_order_used_for_orthogonalization": list(signals),
    }
    return df, all_cluster_orth_cols, meta


def _compute_daily_rank_ic_roll(panel: pd.DataFrame, signal_col: str, horizon: int, roll: int) -> pd.Series:
    """Compute daily cross-sectional rank-IC rolling mean for `signal_col`."""
    if "ret_total" not in panel.columns:
        raise KeyError("Panel must contain 'ret_total' to compute forward returns for IC weights.")

    df = panel[["permno", "date", signal_col, "ret_total"]].dropna().copy()
    df = df.sort_values(["permno", "date"])

    one_plus = 1.0 + df["ret_total"].astype(float)
    fwd = (
        one_plus.groupby(df["permno"]).shift(-1)
        .groupby(df["permno"]).rolling(horizon, min_periods=horizon)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        .shift(-(horizon - 1))
    )
    df["fwd_ret_h"] = fwd - 1.0
    df = df.dropna(subset=["fwd_ret_h"])

    df["sig_rank"] = df.groupby("date")[signal_col].rank(pct=True)
    df["ret_rank"] = df.groupby("date")["fwd_ret_h"].rank(pct=True)

    ic_series = (
        df.groupby("date")[["sig_rank", "ret_rank"]]
        .apply(lambda x: x["sig_rank"].corr(x["ret_rank"]))
        .rename("rank_ic")
        .sort_index()
    )

    ic_roll = ic_series.rolling(roll, min_periods=roll).mean()
    ic_roll.name = "ic_roll"
    return ic_roll


def _find_alpha_validation_dir(alpha_root: Path, signal_name: str) -> Path:
    """Find a matching alpha_validation output directory for a given signal name."""
    candidates = sorted(alpha_root.glob(f"signal={signal_name}__*"))
    if not candidates:
        raise FileNotFoundError(f"No alpha_validation outputs found for signal={signal_name} under {alpha_root}")

    non_reg = [p for p in candidates if "__REG=" not in p.name]
    return non_reg[0] if non_reg else candidates[0]


def _load_ic_series(
    path: Path,
    horizon: int = 5,
    roll: int = 252,
    fallback_panel: pd.DataFrame | None = None,
    fallback_signal_col: str | None = None,
) -> pd.Series:
    """Load ic_series.csv and return rolling mean IC (rank-IC preferred) for a given horizon."""
    ic_path = path / "ic_series.csv"
    if not ic_path.exists():
        raise FileNotFoundError(f"Missing ic_series.csv at {ic_path}")

    ic = pd.read_csv(ic_path, parse_dates=["date"])
    cols = set(ic.columns)

    if ("h" in cols) or ("horizon" in cols):
        hcol = "h" if "h" in cols else "horizon"
        ic = ic.sort_values("date")
        ic = ic[ic[hcol] == horizon].copy()
        if ic.empty:
            raise ValueError(f"No rows found for horizon={horizon} in {ic_path}")
    else:
        rolling_path = path / "rolling_ic.csv"
        if rolling_path.exists():
            ric = pd.read_csv(rolling_path, parse_dates=["date"])
            rcols = set(ric.columns)
            if ("h" in rcols) or ("horizon" in rcols):
                hcol = "h" if "h" in rcols else "horizon"
                ric = ric.sort_values("date")
                ric = ric[ric[hcol] == horizon].copy()
                if ric.empty:
                    raise ValueError(f"No rows found for horizon={horizon} in {rolling_path}")
                ic = ric
                cols = set(ic.columns)
            else:
                if (fallback_panel is None) or (fallback_signal_col is None):
                    raise KeyError(
                        f"No horizon column in {ic_path} or {rolling_path}. To use rolling IC weights for H={horizon}, "
                        f"rerun alpha_validation to export per-horizon IC series, or pass fallback_panel to compute IC directly."
                    )
                return _compute_daily_rank_ic_roll(fallback_panel, fallback_signal_col, horizon=horizon, roll=roll)
        else:
            if (fallback_panel is None) or (fallback_signal_col is None):
                raise KeyError(
                    f"Could not find horizon column ('h' or 'horizon') in {ic_path} and no rolling_ic.csv present."
                )
            return _compute_daily_rank_ic_roll(fallback_panel, fallback_signal_col, horizon=horizon, roll=roll)

    preferred = ["rank_ic", "ic_rank", "rankic", "ic"]
    ic_col = next((c for c in preferred if c in cols), None)
    if ic_col is None:
        ic_like = [c for c in ic.columns if "ic" in c.lower()]
        if not ic_like:
            raise KeyError(f"No IC-like column found in {ic_path}. Columns: {list(ic.columns)}")
        ic_col = ic_like[0]

    ic["ic_roll"] = ic[ic_col].rolling(roll, min_periods=roll).mean()
    return ic.set_index("date")["ic_roll"]


def _normalize_weights(W: pd.DataFrame) -> pd.DataFrame:
    """Normalize weights by sum(|w|) each date. If denom=0, weights become NaN."""
    denom = W.abs().sum(axis=1)
    return W.div(denom.replace(0.0, np.nan), axis=0)


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


def main(
    horizon: int = 5,
    roll: int = 252,
    signals: list[str] | None = None,
    exclude_signals: str = "",
    liquidity_filter_type: str = "ADV",
    liquidity_top_n: int = 1000,
    liquidity_recon_freq: str = "M",
    adv_window: int = 20,
    data_path: str = "data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
    out_dir_str: str = "data/outputs/multi_signal",
    orth: bool = False,
    cluster_orth: bool = False,
    cluster_defs: list[str] | None = None,
    rebalance_frequency: str = "W",
    rebalance_day: str = "FRI",
    orth_threshold: float = 0.20,
    alpha_root_str: str = "data/outputs/alpha_validation",
):
    if not signals:
        raise ValueError("You must provide at least one signal via --signals.")

    if orth and cluster_orth:
        raise ValueError("Use only one orthogonalization mode: --orth or --cluster-orth, not both.")
    if cluster_orth and not cluster_defs:
        raise ValueError("--cluster-orth requires --clusters.")

    panel = pd.read_parquet(data_path)
    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    # Use existing signal columns when available; compute only missing ones.
    panel, signals = prepare_signals(panel, list(signals))

    excl = [s.strip() for s in str(exclude_signals).split(",") if s.strip()]
    if excl:
        signals = [s for s in signals if s not in set(excl)]
        if len(signals) < 2:
            raise ValueError(f"After excluding {excl}, fewer than 2 signals remain: {signals}")

    univ_spec = UniverseSpec(
        filter_type=str(liquidity_filter_type),
        top_n=int(liquidity_top_n),
        recon_freq=str(liquidity_recon_freq),
        adv_window=int(adv_window),
    )
    mask = build_liquidity_universe_mask(panel, univ_spec)
    panel = apply_universe_mask(panel, mask)

    daily_panel = panel.copy()
    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(daily_panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(rebalance_frequency),
            "REBALANCE_DAY": str(rebalance_day),
        },
    )
    panel = daily_panel[daily_panel["date"].isin(rebalance_dates)].copy()

    panel = panel.dropna(subset=signals).copy()

    orth_meta = None
    if orth:
        panel, orth_cols, orth_meta = orthogonalize_signals_rank_space(
            panel,
            signals=signals,
            abs_corr_threshold=float(orth_threshold),
        )
        rank_cols = list(orth_cols)
    elif cluster_orth:
        panel, orth_cols, orth_meta = cluster_orthogonalize_signals_rank_space(
            panel,
            signals=signals,
            cluster_args=list(cluster_defs or []),
            abs_corr_threshold=float(orth_threshold),
        )
        rank_cols = list(orth_cols)
    else:
        for s in signals:
            panel[s + "_rank"] = panel.groupby("date")[s].rank(pct=True)
        rank_cols = [s + "_rank" for s in signals]

    alpha_root = Path(alpha_root_str)

    # Use signal names directly for alpha_validation lookup.
    weights = {}
    for s in signals:
        out_dir = _find_alpha_validation_dir(alpha_root, s)
        w = _load_ic_series(out_dir, horizon=horizon, roll=roll, fallback_panel=daily_panel, fallback_signal_col=s)
        weights[s] = w

    W = pd.DataFrame(weights).sort_index()

    dates = pd.Index(sorted(panel["date"].unique()), name="date")
    W = W.reindex(dates).ffill()
    Wn = _normalize_weights(W)

    panel = panel.copy()
    for s in signals:
        panel = panel.merge(
            Wn[[s]].rename(columns={s: s + "_w"}),
            left_on="date",
            right_index=True,
            how="left",
        )

    if orth:
        sig_to_rank = {s: f"{s}_orth_rank" for s in signals}
    elif cluster_orth:
        sig_to_rank = {s: f"{s}_cluster_orth_rank" for s in signals}
    else:
        sig_to_rank = {s: f"{s}_rank" for s in signals}

    panel["composite_icw_rank"] = 0.0
    for s in signals:
        panel["composite_icw_rank"] += panel[s + "_w"] * panel[sig_to_rank[s]]

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_tag = _safe_tag("-".join(signals))
    rebalance_tag = _safe_tag(f"{rebalance_frequency}-{rebalance_day}")

    tag = f"__SIG={signal_tag}__RB={rebalance_tag}"
    if orth:
        tag += f"__ORTH_thr{orth_threshold:.2f}"
    elif cluster_orth:
        cluster_tag = _safe_tag("-".join(cluster_defs or []))
        tag += f"__CLUSTER_ORTH_thr{orth_threshold:.2f}__CL={cluster_tag}"
    if excl:
        safe = _safe_tag("-".join(excl))
        tag += f"__EXCL={safe}"

    Wn.to_csv(out_dir / f"ic_weights_roll{roll}_H{horizon}{tag}.csv")

    keep = ["permno", "date", "composite_icw_rank"] + rank_cols
    panel[keep].to_parquet(out_dir / f"composite_icw_rank_roll{roll}_H{horizon}{tag}.parquet", index=False)

    meta = {
        "data": str(data_path),
        "alpha_root": str(alpha_root_str),
        "horizon": horizon,
        "roll": roll,
        "signals": signals,
        "exclude_signals": excl,
        "orth": bool(orth),
        "cluster_orth": bool(cluster_orth),
        "cluster_defs": list(cluster_defs or []),
        "orth_threshold": float(orth_threshold),
        "orth_meta": orth_meta,
        "rank_cols": rank_cols,
        "weight_source": "data/outputs/alpha_validation/*/ic_series.csv",
        "weight_method": "rolling_mean_ic",
        "weight_normalization": "w / sum(|w|) per date",
        "composite": "sum_j w_j(t) * rank_j(i,t)",
        "rebalance_frequency": str(rebalance_frequency),
        "rebalance_day": str(rebalance_day),
        "universe": {
            "type": str(liquidity_filter_type),
            "top_n": int(liquidity_top_n),
            "recon_freq": str(liquidity_recon_freq),
            "adv_window": int(adv_window),
            "source": "backtest.universe",
        },
    }
    with open(out_dir / f"composite_icw_rank_roll{roll}_H{horizon}{tag}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved composite + weights to: {out_dir}")
    print(f"Composite file: composite_icw_rank_roll{roll}_H{horizon}{tag}.parquet")
    print(f"Weights file:   ic_weights_roll{roll}_H{horizon}{tag}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build IC-weighted composite signal (rank-space).")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading days (default 5).")
    parser.add_argument("--roll", type=int, default=252, help="Rolling window length for IC weights (default 252).")
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Signal columns to use. Existing panel columns are used directly; missing ones are computed.",
    )
    parser.add_argument("--orth", action="store_true", help="Orthogonalize signals before IC-weighting (rank-space).")
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="W",
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
        help="Rebalance frequency used to define the composite research grid.",
    )
    parser.add_argument(
        "--rebalance-day",
        type=str,
        default="FRI",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        help="Weekday used when rebalance frequency is W/BW/TW.",
    )
    parser.add_argument(
        "--cluster-orth",
        action="store_true",
        help="Orthogonalize signals only within specified clusters before IC-weighting.",
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
    parser.add_argument("--orth_threshold", type=float, default=0.20, help="Orth correlation threshold (default 0.20).")
    parser.add_argument(
        "--exclude_signals",
        type=str,
        default="",
        help="Comma-separated signals to exclude (e.g. 'idiosyncratic_vol' or 'beta_sig,momentum12_1').",
    )
    parser.add_argument("--liquidity_filter_type", type=str, default="ADV", help="Universe filter type (default ADV).")
    parser.add_argument("--liquidity_top_n", type=int, default=1000, help="Universe size N (default 1000).")
    parser.add_argument("--liquidity_recon_freq", type=str, default="M", help="Universe recon frequency (default M).")
    parser.add_argument("--adv_window", type=int, default=20, help="ADV lookback window (default 20).")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet",
        help="Input daily panel parquet path.",
    )
    parser.add_argument(
        "--alpha_root",
        type=str,
        default="data/outputs/alpha_validation",
        help="Root directory containing alpha_validation outputs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/outputs/multi_signal",
        help="Output directory for composite artifacts.",
    )
    args = parser.parse_args()

    main(
        horizon=args.horizon,
        roll=args.roll,
        signals=args.signals,
        exclude_signals=args.exclude_signals,
        liquidity_filter_type=args.liquidity_filter_type,
        liquidity_top_n=args.liquidity_top_n,
        liquidity_recon_freq=args.liquidity_recon_freq,
        adv_window=args.adv_window,
        data_path=args.data,
        out_dir_str=args.out_dir,
        orth=args.orth,
        cluster_orth=args.cluster_orth,
        cluster_defs=args.clusters,
        rebalance_frequency=args.rebalance_frequency,
        rebalance_day=args.rebalance_day,
        orth_threshold=args.orth_threshold,
        alpha_root_str=args.alpha_root,
    )