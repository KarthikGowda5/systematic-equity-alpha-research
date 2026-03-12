

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type for {path}")


def _ensure_datetime(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _find_first_existing(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {label}. Tried columns: {list(candidates)}")


def _safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num_s = pd.Series(num)
    den_s = pd.Series(den)
    out = pd.Series(np.zeros(len(num_s), dtype=float), index=num_s.index)
    mask = den_s.abs() > 1e-12
    out.loc[mask] = num_s.loc[mask] / den_s.loc[mask]
    return out


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class AttributionInputs:
    run_dir: Path
    panel_path: Path
    factor_path: Path
    signal_weights_path: Optional[Path] = None


@dataclass
class RunOutputs:
    weights: pd.DataFrame
    period_returns: pd.DataFrame
    period_details: pd.DataFrame
    turnover: pd.DataFrame
    metrics: Dict
    settings: Dict


@dataclass
class AttributionResults:
    attribution_daily: pd.DataFrame
    factor_attribution: pd.DataFrame
    sector_attribution: pd.DataFrame
    cost_attribution: pd.DataFrame
    sleeve_attribution: pd.DataFrame
    summary: Dict


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def load_run_outputs(run_dir: str | Path) -> RunOutputs:
    run_dir = Path(run_dir)

    weights = _read_table(run_dir / "weights.csv")
    period_returns = _read_table(run_dir / "period_returns.csv")
    turnover = _read_table(run_dir / "turnover.csv")
    period_details_path = run_dir / "period_details.csv"
    if period_details_path.exists():
        period_details = _read_table(period_details_path)
    else:
        period_details = pd.DataFrame()

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(run_dir / "settings_used.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    weights = _ensure_datetime(weights, ["date", "rebalance_date"])
    period_returns = _ensure_datetime(period_returns, ["date", "rebalance_date"])
    turnover = _ensure_datetime(turnover, ["date", "rebalance_date"])
    period_details = _ensure_datetime(period_details, ["date", "rebalance_date", "formation_date"])

    return RunOutputs(
        weights=weights,
        period_returns=period_returns,
        period_details=period_details,
        turnover=turnover,
        metrics=metrics,
        settings=settings,
    )


def load_panel(panel_path: str | Path) -> pd.DataFrame:
    panel = _read_table(panel_path)
    panel = _ensure_datetime(panel, ["date", "rebalance_date"])
    return panel


def load_factor_data(factor_path: str | Path) -> pd.DataFrame:
    ff = _read_table(factor_path)
    ff = _ensure_datetime(ff, ["date", "Date"])

    date_col = _find_first_existing(ff, ["date", "Date"], "factor date column")
    ff = ff.rename(columns={date_col: "date"}).copy()

    rename_map = {
        "Mkt-RF": "mkt_excess",
        "MKT_RF": "mkt_excess",
        "SMB": "smb",
        "HML": "hml",
        "RMW": "rmw",
        "CMA": "cma",
        "Mom": "mom",
        "MOM": "mom",
        "UMD": "mom",
        "RF": "rf",
    }
    ff = ff.rename(columns={k: v for k, v in rename_map.items() if k in ff.columns})

    for col in ["mkt_excess", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        if col in ff.columns:
            ff[col] = pd.to_numeric(ff[col], errors="coerce")
            # Fama-French files are usually in percent.
            if ff[col].dropna().abs().median() > 0.2:
                ff[col] = ff[col] / 100.0

    keep_cols = [c for c in ["date", "mkt_excess", "smb", "hml", "rmw", "cma", "mom", "rf"] if c in ff.columns]
    return ff[keep_cols].sort_values("date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Core preparation
# -----------------------------------------------------------------------------


def prepare_holdings_frame(weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    permno_col_w = _find_first_existing(weights, ["permno", "PERMNO"], "weights permno column")
    date_col_w = _find_first_existing(weights, ["date", "rebalance_date"], "weights date column")
    weight_col = _find_first_existing(weights, ["weight", "target_weight", "post_cost_weight"], "weight column")

    permno_col_p = _find_first_existing(panel, ["permno", "PERMNO"], "panel permno column")
    date_col_p = _find_first_existing(panel, ["date"], "panel date column")
    ret_col = _find_first_existing(
        panel,
        ["ret", "retx", "return", "dlret_total", "ret_total", "total_return", "retadj"],
        "panel return column",
    )

    sector_col = _find_first_existing(panel, ["sector", "sector_name", "gics_sector"], "sector column")

    cols = [permno_col_p, date_col_p, ret_col, sector_col]
    optional_cols = [
        "beta",
        "composite_raw",
        "composite_raw_smooth",
        "composite_orth",
        "composite_raw_no_idio",
        "composite_orth_no_idio",
        "STR",
        "Residual_mom",
        "Volatility_AR",
        "idiosyncratic_vol",
        "beta_signal",
        "momentum12_1",
        "mom_12_1",
    ]
    cols.extend([c for c in optional_cols if c in panel.columns])

    panel_small = panel[cols].copy()
    panel_small = panel_small.rename(
        columns={
            permno_col_p: "permno",
            date_col_p: "date",
            ret_col: "asset_return",
            sector_col: "sector",
        }
    )

    weights_small = weights[[permno_col_w, date_col_w, weight_col]].copy()
    weights_small = weights_small.rename(
        columns={permno_col_w: "permno", date_col_w: "date", weight_col: "weight"}
    )

    holdings = weights_small.merge(panel_small, on=["permno", "date"], how="left", validate="many_to_one")
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce").fillna(0.0)
    holdings["asset_return"] = pd.to_numeric(holdings["asset_return"], errors="coerce").fillna(0.0)
    holdings["pnl_contrib"] = holdings["weight"] * holdings["asset_return"]
    holdings["abs_weight"] = holdings["weight"].abs()

    return holdings.sort_values(["date", "permno"]).reset_index(drop=True)


def compute_daily_portfolio_returns_from_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    daily = (
        holdings.groupby("date", as_index=False)
        .agg(
            gross_return=("pnl_contrib", "sum"),
            gross_exposure=("abs_weight", "sum"),
            net_exposure=("weight", "sum"),
            n_names=("permno", "nunique"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


# -----------------------------------------------------------------------------
# Attribution blocks
# -----------------------------------------------------------------------------


def compute_factor_attribution(daily_returns: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = daily_returns.merge(factors, on="date", how="left")

    available_factor_cols = [c for c in ["mkt_excess", "smb", "hml", "rmw", "cma", "mom"] if c in df.columns]
    if not available_factor_cols:
        out = df[["date", "gross_return"]].copy()
        out["explained_factor_return"] = 0.0
        out["alpha_residual"] = out["gross_return"]
        return out, pd.DataFrame(columns=["factor", "beta", "mean_factor_return", "total_contribution"])

    reg_df = df[["gross_return"] + available_factor_cols].dropna().copy()
    y = reg_df["gross_return"].to_numpy(dtype=float)
    x = reg_df[available_factor_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])

    coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    intercept = float(coefs[0])
    betas = dict(zip(available_factor_cols, coefs[1:]))

    out = df[["date", "gross_return"] + available_factor_cols].copy()
    for factor in available_factor_cols:
        out[f"{factor}_contrib"] = betas[factor] * out[factor].fillna(0.0)

    contrib_cols = [f"{f}_contrib" for f in available_factor_cols]
    out["intercept_alpha"] = intercept
    out["explained_factor_return"] = out[contrib_cols].sum(axis=1)
    out["alpha_residual"] = out["gross_return"] - out["explained_factor_return"]

    factor_summary = pd.DataFrame(
        {
            "factor": ["intercept"] + available_factor_cols,
            "beta": [intercept] + [float(betas[f]) for f in available_factor_cols],
            "mean_factor_return": [np.nan]
            + [float(out[f].dropna().mean()) for f in available_factor_cols],
            "total_contribution": [float(intercept * len(out))]
            + [float(out[f"{f}_contrib"].sum()) for f in available_factor_cols],
        }
    )

    keep_cols = [
        "date",
        "gross_return",
        *available_factor_cols,
        *contrib_cols,
        "intercept_alpha",
        "explained_factor_return",
        "alpha_residual",
    ]
    return out[keep_cols], factor_summary


def compute_sector_attribution(holdings: pd.DataFrame) -> pd.DataFrame:
    by_sector_day = (
        holdings.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_pnl=("pnl_contrib", "sum"),
            sector_net_weight=("weight", "sum"),
            sector_gross_weight=("abs_weight", "sum"),
        )
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )

    summary = (
        by_sector_day.groupby("sector", as_index=False)
        .agg(
            total_sector_pnl=("sector_pnl", "sum"),
            avg_net_weight=("sector_net_weight", "mean"),
            avg_gross_weight=("sector_gross_weight", "mean"),
            n_days=("date", "nunique"),
        )
        .sort_values("total_sector_pnl", ascending=False)
        .reset_index(drop=True)
    )

    total_pnl = summary["total_sector_pnl"].sum()
    summary["pct_of_total_pnl"] = 0.0 if abs(total_pnl) < 1e-12 else summary["total_sector_pnl"] / total_pnl
    return summary


def compute_cost_attribution(
    period_details: pd.DataFrame,
    daily_returns: pd.DataFrame,
    period_returns: pd.DataFrame,
    turnover: pd.DataFrame,
) -> pd.DataFrame:
    if period_details is not None and not period_details.empty:
        details = period_details.copy()
        details = _ensure_datetime(details, ["date", "formation_date"])

        rename_map = {}
        if "total_cost" in details.columns and "total_cost_drag" not in details.columns:
            rename_map["total_cost"] = "total_cost_drag"
        details = details.rename(columns=rename_map)

        keep_cols = [
            c
            for c in [
                "formation_date",
                "date",
                "gross_return",
                "net_return",
                "turnover",
                "spread_cost",
                "vol_slippage_cost",
                "turnover_cost",
                "total_cost_drag",
                "gross_exposure_used",
                "net_exposure_used",
                "n_holdings",
                "participation_constrained",
            ]
            if c in details.columns
        ]
        out = details[keep_cols].copy()

        numeric_cols = [
            "gross_return",
            "net_return",
            "turnover",
            "spread_cost",
            "vol_slippage_cost",
            "turnover_cost",
            "total_cost_drag",
            "gross_exposure_used",
            "net_exposure_used",
            "n_holdings",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        if "total_cost_drag" not in out.columns and {"gross_return", "net_return"}.issubset(out.columns):
            out["total_cost_drag"] = out["gross_return"] - out["net_return"]

        return out.sort_values("date").reset_index(drop=True)

    pr = period_returns.copy()
    pr = _ensure_datetime(pr, ["date", "rebalance_date"])

    date_col = _find_first_existing(pr, ["date", "rebalance_date"], "period return date column")
    pr = pr.rename(columns={date_col: "date"})

    gross_col_candidates = ["gross_return", "portfolio_return_gross", "return_gross"]
    net_col_candidates = ["net_return", "portfolio_return_net", "return_net", "period_return"]

    gross_col = next((c for c in gross_col_candidates if c in pr.columns), None)
    net_col = next((c for c in net_col_candidates if c in pr.columns), None)

    out = daily_returns[["date", "gross_return"]].copy()

    if gross_col is not None:
        pr[gross_col] = pd.to_numeric(pr[gross_col], errors="coerce")
        out = out.merge(pr[["date", gross_col]], on="date", how="left", suffixes=("", "_reported"))
        out["gross_return_reported"] = out[gross_col]
        out = out.drop(columns=[gross_col])
    else:
        out["gross_return_reported"] = np.nan

    if net_col is not None:
        pr[net_col] = pd.to_numeric(pr[net_col], errors="coerce")
        out = out.merge(pr[["date", net_col]], on="date", how="left")
        out = out.rename(columns={net_col: "net_return"})
    else:
        out["net_return"] = np.nan

    turnover_df = turnover.copy()
    date_col_to = _find_first_existing(turnover_df, ["date", "rebalance_date"], "turnover date column")
    turnover_df = turnover_df.rename(columns={date_col_to: "date"})
    to_col = _find_first_existing(turnover_df, ["turnover", "portfolio_turnover"], "turnover column")
    turnover_df[to_col] = pd.to_numeric(turnover_df[to_col], errors="coerce")
    out = out.merge(turnover_df[["date", to_col]], on="date", how="left")
    out = out.rename(columns={to_col: "turnover"})

    out["total_cost_drag"] = out["gross_return"] - out["net_return"]
    out["spread_cost"] = np.nan
    out["vol_slippage_cost"] = np.nan
    out["turnover_cost"] = np.nan

    keep_cols = [
        "date",
        "gross_return",
        "net_return",
        "turnover",
        "total_cost_drag",
        "spread_cost",
        "vol_slippage_cost",
        "turnover_cost",
    ]
    return out[keep_cols].sort_values("date").reset_index(drop=True)


def _load_signal_weight_history(signal_weights_path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if signal_weights_path is None:
        return None
    df = _read_table(signal_weights_path)
    df = _ensure_datetime(df, ["date", "rebalance_date"])
    date_col = _find_first_existing(df, ["date", "rebalance_date"], "signal weights date column")
    df = df.rename(columns={date_col: "date"}).copy()
    return df


def compute_sleeve_attribution(
    holdings: pd.DataFrame,
    signal_weights_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    signal_weight_history = _load_signal_weight_history(signal_weights_path)

    candidate_sleeves = [
        "STR",
        "Residual_mom",
        "Volatility_AR",
        "idiosyncratic_vol",
        "beta",
        "momentum12_1",
        "mom_12_1",
    ]
    sleeve_cols = [c for c in candidate_sleeves if c in holdings.columns]
    if not sleeve_cols:
        return pd.DataFrame(columns=["sleeve", "total_pnl", "pct_of_total_pnl"])

    work = holdings[["date", "permno", "pnl_contrib"] + sleeve_cols].copy()

    if signal_weight_history is not None:
        join_cols = [c for c in sleeve_cols if c in signal_weight_history.columns]
        if join_cols:
            sw = signal_weight_history[["date"] + join_cols].copy()
            sw = sw.rename(columns={c: f"{c}__combo_w" for c in join_cols})
            work = work.merge(sw, on="date", how="left")
            for col in join_cols:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0) * pd.to_numeric(
                    work[f"{col}__combo_w"], errors="coerce"
                ).fillna(1.0)

    for col in sleeve_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    abs_sum = work[sleeve_cols].abs().sum(axis=1)
    for col in sleeve_cols:
        phi_col = f"{col}__phi"
        work[phi_col] = np.where(abs_sum > 1e-12, work[col].abs() / abs_sum, 0.0)
        work[f"{col}__pnl"] = work["pnl_contrib"] * work[phi_col]

    sleeve_pnl_cols = [f"{c}__pnl" for c in sleeve_cols]
    daily = (
        work.groupby("date", as_index=False)[sleeve_pnl_cols]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    long = daily.melt(id_vars="date", var_name="sleeve", value_name="daily_pnl")
    long["sleeve"] = long["sleeve"].str.replace("__pnl", "", regex=False)
    long["cumulative_pnl"] = long.groupby("sleeve")["daily_pnl"].cumsum()

    totals = long.groupby("sleeve", as_index=False)["daily_pnl"].sum().rename(columns={"daily_pnl": "total_pnl"})
    total_all = totals["total_pnl"].sum()
    totals["pct_of_total_pnl"] = 0.0 if abs(total_all) < 1e-12 else totals["total_pnl"] / total_all

    long = long.merge(totals, on="sleeve", how="left")
    return long.sort_values(["sleeve", "date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Summary + memo
# -----------------------------------------------------------------------------


def build_summary(
    daily_returns: pd.DataFrame,
    factor_attribution: pd.DataFrame,
    sector_attribution: pd.DataFrame,
    cost_attribution: pd.DataFrame,
    sleeve_attribution: pd.DataFrame,
    metrics: Optional[Mapping] = None,
    settings: Optional[Mapping] = None,
) -> Dict:
    total_gross = float(daily_returns["gross_return"].sum())
    total_net = float(cost_attribution["net_return"].dropna().sum()) if "net_return" in cost_attribution else np.nan
    total_cost = float(cost_attribution["total_cost_drag"].dropna().sum()) if "total_cost_drag" in cost_attribution else np.nan
    total_factor = float(factor_attribution["explained_factor_return"].sum()) if "explained_factor_return" in factor_attribution else 0.0
    total_alpha = float(factor_attribution["alpha_residual"].sum()) if "alpha_residual" in factor_attribution else total_gross

    top_sector = None
    worst_sector = None
    if not sector_attribution.empty:
        top_sector = sector_attribution.sort_values("total_sector_pnl", ascending=False).iloc[0]["sector"]
        worst_sector = sector_attribution.sort_values("total_sector_pnl", ascending=True).iloc[0]["sector"]

    top_sleeve = None
    if not sleeve_attribution.empty:
        sleeve_totals = (
            sleeve_attribution[["sleeve", "total_pnl"]]
            .drop_duplicates()
            .sort_values("total_pnl", ascending=False)
            .reset_index(drop=True)
        )
        if not sleeve_totals.empty:
            top_sleeve = sleeve_totals.iloc[0]["sleeve"]

    return {
        "total_gross_return": total_gross,
        "total_net_return": total_net,
        "total_cost_drag": total_cost,
        "total_factor_explained_return": total_factor,
        "total_alpha_residual": total_alpha,
        "top_sector": top_sector,
        "worst_sector": worst_sector,
        "top_sleeve": top_sleeve,
        "metrics": dict(metrics or {}),
        "settings": dict(settings or {}),
    }


def build_memo_text(summary: Mapping, factor_attribution: pd.DataFrame, sector_attribution: pd.DataFrame, sleeve_attribution: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("## Performance Attribution")
    lines.append("")
    lines.append("### Overall Return Decomposition")
    lines.append(f"- Total gross return: {summary.get('total_gross_return', np.nan):.6f}")
    lines.append(f"- Total net return: {summary.get('total_net_return', np.nan):.6f}")
    lines.append(f"- Total transaction cost drag: {summary.get('total_cost_drag', np.nan):.6f}")
    lines.append("")
    lines.append("### Factor Exposure vs Alpha")
    lines.append(f"- Factor-explained return: {summary.get('total_factor_explained_return', np.nan):.6f}")
    lines.append(f"- Residual alpha: {summary.get('total_alpha_residual', np.nan):.6f}")
    lines.append(
        f"- Interpretation: realized PnL split between explained factor return ({summary.get('total_factor_explained_return', np.nan):.6f}) and residual alpha ({summary.get('total_alpha_residual', np.nan):.6f})."
    )
    lines.append("")
    lines.append("### Sector Contribution")
    if sector_attribution.empty:
        lines.append("- No sector attribution available.")
    else:
        top = sector_attribution.sort_values("total_sector_pnl", ascending=False).head(3)
        bottom = sector_attribution.sort_values("total_sector_pnl", ascending=True).head(3)
        lines.append("- Top contributing sectors:")
        for _, row in top.iterrows():
            lines.append(f"  - {row['sector']}: {row['total_sector_pnl']:.6f}")
        lines.append("- Biggest detracting sectors:")
        for _, row in bottom.iterrows():
            lines.append(f"  - {row['sector']}: {row['total_sector_pnl']:.6f}")
    lines.append("")
    lines.append("### Signal Sleeve Attribution")
    if sleeve_attribution.empty:
        lines.append("- No sleeve attribution available.")
    else:
        sleeve_totals = (
            sleeve_attribution[["sleeve", "total_pnl", "pct_of_total_pnl"]]
            .drop_duplicates()
            .sort_values("total_pnl", ascending=False)
        )
        for _, row in sleeve_totals.iterrows():
            lines.append(f"- {row['sleeve']}: total_pnl={row['total_pnl']:.6f}, pct_of_total_pnl={row['pct_of_total_pnl']:.2%}")
    lines.append("")
    lines.append("### Main Drivers")
    lines.append(
        f"- Primary positive drivers appear to be residual alpha and the strongest sleeve / sectors identified above."
    )
    lines.append(
        f"- Transaction costs reduced realized performance by {summary.get('total_cost_drag', np.nan):.6f} over the sample."
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_attribution(inputs: AttributionInputs) -> AttributionResults:
    run_outputs = load_run_outputs(inputs.run_dir)
    panel = load_panel(inputs.panel_path)
    factors = load_factor_data(inputs.factor_path)

    holdings = prepare_holdings_frame(run_outputs.weights, panel)
    daily_returns = compute_daily_portfolio_returns_from_weights(holdings)
    factor_daily, factor_summary = compute_factor_attribution(daily_returns, factors)
    sector_summary = compute_sector_attribution(holdings)
    cost_daily = compute_cost_attribution(
        run_outputs.period_details,
        daily_returns,
        run_outputs.period_returns,
        run_outputs.turnover,
    )
    sleeve_daily = compute_sleeve_attribution(holdings, inputs.signal_weights_path)

    attribution_daily = daily_returns.merge(
        factor_daily[[c for c in factor_daily.columns if c != "gross_return"]], on="date", how="left"
    ).merge(
        cost_daily[[c for c in cost_daily.columns if c not in {"gross_return"}]], on="date", how="left"
    )

    summary = build_summary(
        daily_returns=daily_returns,
        factor_attribution=factor_daily,
        sector_attribution=sector_summary,
        cost_attribution=cost_daily,
        sleeve_attribution=sleeve_daily,
        metrics=run_outputs.metrics,
        settings=run_outputs.settings,
    )

    return AttributionResults(
        attribution_daily=attribution_daily,
        factor_attribution=factor_summary,
        sector_attribution=sector_summary,
        cost_attribution=cost_daily,
        sleeve_attribution=sleeve_daily,
        summary=summary,
    )


def write_outputs(results: AttributionResults, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results.attribution_daily.to_csv(out_dir / "attribution_daily.csv", index=False)
    results.factor_attribution.to_csv(out_dir / "factor_attribution.csv", index=False)
    results.sector_attribution.to_csv(out_dir / "sector_attribution.csv", index=False)
    results.cost_attribution.to_csv(out_dir / "cost_attribution.csv", index=False)
    results.sleeve_attribution.to_csv(out_dir / "sleeve_attribution.csv", index=False)

    with open(out_dir / "attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(results.summary, f, indent=2)

    memo = build_memo_text(
        results.summary,
        results.factor_attribution,
        results.sector_attribution,
        results.sleeve_attribution,
    )
    with open(out_dir / "attribution_memo.md", "w", encoding="utf-8") as f:
        f.write(memo)