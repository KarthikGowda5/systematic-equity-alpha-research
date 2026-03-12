"""Stress testing utilities (Phase 10).

Implements:
- Fixed crisis windows (2008 GFC, 2020 COVID crash)
- High-volatility regime defined from the strategy's own realized volatility

Inputs:
- returns: pd.Series of periodic portfolio returns (typically daily), indexed by date

Outputs:
- pandas DataFrame with per-regime summary metrics

Design notes:
- We intentionally avoid external dependencies like VIX to keep this phase
  reproducible given the current repository data.
- The high-vol regime is defined using rolling annualized vol of the
  portfolio return series, selecting the top quantile of dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .metrics import MetricConfig, compute_standard_metrics, rolling_vol


@dataclass(frozen=True)
class StressConfig:
    ann_factor: int = 252
    rf_annual: float = 0.0
    mar_annual: float = 0.0

    # High-vol regime definition
    hv_window: int = 20
    hv_top_quantile: float = 0.90  # top 10% by rolling vol

    # Crisis windows (inclusive bounds)
    gfc_start: str = "2008-09-01"
    gfc_end: str = "2009-03-31"

    covid_start: str = "2020-02-15"
    covid_end: str = "2020-05-01"


def _to_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    r = returns.copy()
    if not isinstance(r.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            r.index = pd.to_datetime(r.index)
        except Exception as e:
            raise TypeError("returns index must be datetime-like or convertible") from e

    r = pd.to_numeric(r, errors="coerce").astype(float).dropna().sort_index()
    return r


def _slice_by_date(r: pd.Series, start: str, end: str) -> pd.Series:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return r.loc[(r.index >= start_ts) & (r.index <= end_ts)]


def _summarize_regime(r: pd.Series, cfg: StressConfig) -> Dict[str, float]:
    mc = MetricConfig(ann_factor=cfg.ann_factor, rf_annual=cfg.rf_annual, mar_annual=cfg.mar_annual)
    metrics, _, _ = compute_standard_metrics(r, mc)

    out: Dict[str, float] = {
        "n_obs": float(metrics.get("n_obs", float("nan"))),
        "ann_return": float(metrics.get("ann_return", float("nan"))),
        "ann_vol": float(metrics.get("ann_vol", float("nan"))),
        "sharpe": float(metrics.get("sharpe", float("nan"))),
        "sortino": float(metrics.get("sortino", float("nan"))),
        "max_drawdown": float(metrics.get("max_drawdown", float("nan"))),
    }

    if r.size > 0:
        out["mean_return"] = float(r.mean())
        out["std_return"] = float(r.std(ddof=1)) if r.size >= 2 else float("nan")

    return out


def compute_stress_tests(returns: pd.Series, config: Optional[StressConfig] = None) -> pd.DataFrame:
    """Compute stress test metrics for predefined regimes."""
    cfg = config or StressConfig()
    r = _to_series(returns)

    rows: List[Dict[str, float]] = []
    idx: List[str] = []

    # Full sample baseline
    idx.append("full_sample")
    rows.append(_summarize_regime(r, cfg))

    # Crisis windows
    gfc = _slice_by_date(r, cfg.gfc_start, cfg.gfc_end)
    idx.append("gfc_2008")
    rows.append(_summarize_regime(gfc, cfg))

    covid = _slice_by_date(r, cfg.covid_start, cfg.covid_end)
    idx.append("covid_2020")
    rows.append(_summarize_regime(covid, cfg))

    # High-vol regime based on portfolio realized vol
    rv = rolling_vol(r, window=cfg.hv_window, ann_factor=cfg.ann_factor)
    if rv.dropna().size > 0:
        thr = float(rv.quantile(cfg.hv_top_quantile))
        hv_dates = rv[rv >= thr].dropna().index
        hv = r.loc[r.index.isin(hv_dates)]
    else:
        hv = r.iloc[0:0]

    idx.append(f"high_vol_top_{int((1 - cfg.hv_top_quantile) * 100):02d}pct")
    rows.append(_summarize_regime(hv, cfg))

    out = pd.DataFrame(rows, index=idx)

    out["start"] = [
        r.index.min() if not r.empty else pd.NaT,
        pd.to_datetime(cfg.gfc_start),
        pd.to_datetime(cfg.covid_start),
        hv.index.min() if not hv.empty else pd.NaT,
    ]
    out["end"] = [
        r.index.max() if not r.empty else pd.NaT,
        pd.to_datetime(cfg.gfc_end),
        pd.to_datetime(cfg.covid_end),
        hv.index.max() if not hv.empty else pd.NaT,
    ]

    preferred = [
        "start",
        "end",
        "n_obs",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "mean_return",
        "std_return",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]