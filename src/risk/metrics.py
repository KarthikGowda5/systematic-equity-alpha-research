"""Risk metrics (Phase 10).

Compute standard portfolio risk/performance metrics from a single return series.
This module is analysis-only and does not modify portfolio construction.

Expected input:
- returns: pd.Series of periodic returns (typically daily), indexed by date

Outputs:
- dictionaries of scalar metrics
- helper series (equity curve, drawdown) for plotting/reporting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for annualization and target return."""

    ann_factor: int = 252  # daily -> annual
    rf_annual: float = 0.0  # annual risk-free rate, used for Sharpe
    mar_annual: float = 0.0  # annual minimum acceptable return, used for Sortino


def _to_series(returns: pd.Series) -> pd.Series:
    """Validate and coerce returns to a clean float Series indexed by datetime."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    r = returns.copy()

    # Ensure datetime index when possible
    if not isinstance(r.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            r.index = pd.to_datetime(r.index)
        except Exception as e:
            raise TypeError("returns index must be datetime-like or convertible") from e

    # Drop missing and coerce to float
    r = pd.to_numeric(r, errors="coerce").astype(float)
    r = r.dropna()

    # Sort for deterministic behavior
    r = r.sort_index()

    return r


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Compute cumulative equity curve from periodic returns."""
    r = _to_series(returns)
    return start * (1.0 + r).cumprod()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity curve."""
    if not isinstance(equity, pd.Series):
        raise TypeError("equity must be a pandas Series")
    eq = pd.to_numeric(equity, errors="coerce").astype(float).dropna().sort_index()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (most negative drawdown) computed from returns."""
    eq = equity_curve(returns)
    dd = drawdown_series(eq)
    if dd.empty:
        return float("nan")
    return float(dd.min())


def annualized_vol(returns: pd.Series, ann_factor: int = 252) -> float:
    """Annualized volatility from periodic returns."""
    r = _to_series(returns)
    if r.size < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def annualized_return(returns: pd.Series, ann_factor: int = 252) -> float:
    """Annualized geometric return from periodic returns."""
    r = _to_series(returns)
    if r.empty:
        return float("nan")
    growth = (1.0 + r).prod()
    n = r.shape[0]
    return float(growth ** (ann_factor / n) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    ann_factor: int = 252,
    rf_annual: float = 0.0,
) -> float:
    """Annualized Sharpe ratio using an annual risk-free rate."""
    r = _to_series(returns)
    if r.size < 2:
        return float("nan")

    rf_per = (1.0 + rf_annual) ** (1.0 / ann_factor) - 1.0
    ex = r - rf_per
    vol = ex.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(ann_factor))


def sortino_ratio(
    returns: pd.Series,
    ann_factor: int = 252,
    mar_annual: float = 0.0,
) -> float:
    """Annualized Sortino ratio using an annual minimum acceptable return (MAR)."""
    r = _to_series(returns)
    if r.size < 2:
        return float("nan")

    mar_per = (1.0 + mar_annual) ** (1.0 / ann_factor) - 1.0
    diff = r - mar_per
    downside = diff[diff < 0]
    if downside.size < 2:
        return float("inf") if diff.mean() > 0 else float("nan")

    downside_dev = downside.std(ddof=1)
    if downside_dev == 0 or np.isnan(downside_dev):
        return float("nan")

    return float(diff.mean() / downside_dev * np.sqrt(ann_factor))


def compute_standard_metrics(
    returns: pd.Series,
    config: Optional[MetricConfig] = None,
) -> Tuple[Dict[str, float], pd.Series, pd.Series]:
    """Compute standard risk metrics.

    Returns:
        metrics: dict of scalar metrics
        eq: equity curve series
        dd: drawdown series
    """
    cfg = config or MetricConfig()
    r = _to_series(returns)

    eq = equity_curve(r)
    dd = drawdown_series(eq)

    metrics: Dict[str, float] = {
        "n_obs": float(r.shape[0]),
        "ann_return": annualized_return(r, cfg.ann_factor),
        "ann_vol": annualized_vol(r, cfg.ann_factor),
        "sharpe": sharpe_ratio(r, cfg.ann_factor, cfg.rf_annual),
        "sortino": sortino_ratio(r, cfg.ann_factor, cfg.mar_annual),
        "max_drawdown": float(dd.min()) if not dd.empty else float("nan"),
    }

    if r.size > 0:
        metrics["mean_return"] = float(r.mean())
        metrics["std_return"] = float(r.std(ddof=1)) if r.size >= 2 else float("nan")
        metrics["skew"] = float(r.skew()) if r.size >= 3 else float("nan")
        metrics["kurtosis"] = float(r.kurtosis()) if r.size >= 4 else float("nan")

    return metrics, eq, dd


def rolling_vol(returns: pd.Series, window: int = 20, ann_factor: int = 252) -> pd.Series:
    """Rolling annualized volatility."""
    r = _to_series(returns)
    return r.rolling(window).std(ddof=1) * np.sqrt(ann_factor)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    ann_factor: int = 252,
    rf_annual: float = 0.0,
) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    r = _to_series(returns)
    rf_per = (1.0 + rf_annual) ** (1.0 / ann_factor) - 1.0
    ex = r - rf_per

    mean = ex.rolling(window).mean()
    std = ex.rolling(window).std(ddof=1)

    out = mean / std * np.sqrt(ann_factor)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
