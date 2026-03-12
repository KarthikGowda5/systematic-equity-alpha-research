

"""Performance metrics (Phase 4 baseline).

Keep this module limited to simple, transparent metrics used in the baseline report:
- equity curve from periodic returns
- annualized volatility
- Sharpe ratio (rf=0)
- max drawdown

No transaction costs, no risk model, no attribution.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def equity_curve(period_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Compute equity curve from a series of periodic returns.

    Parameters
    ----------
    period_returns : pd.Series
        Period returns indexed by period end date.
    start_value : float
        Starting equity.

    Returns
    -------
    pd.Series
        Equity curve indexed like period_returns.
    """
    pr = period_returns.astype(float)
    eq = (1.0 + pr).cumprod() * float(start_value)
    eq.name = "equity"
    return eq


def max_drawdown(eq: pd.Series) -> float:
    """Max drawdown from an equity curve (most negative peak-to-trough)."""
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def annualization_factor_from_avg_period_days(avg_trading_days_per_period: float) -> float:
    """Return periods_per_year given avg trading days per period.

    Uses 252 trading days per year.
    """
    d = max(float(avg_trading_days_per_period), 1.0)
    return 252.0 / d


def annualized_vol(period_returns: pd.Series, periods_per_year: float) -> float:
    """Annualized volatility from periodic returns."""
    pr = period_returns.astype(float)
    if len(pr) < 2:
        return 0.0
    sig = float(pr.std(ddof=1))
    return sig * math.sqrt(float(periods_per_year)) if sig > 0 else 0.0


def sharpe_ratio(period_returns: pd.Series, periods_per_year: float) -> float:
    """Sharpe ratio (rf=0) from periodic returns."""
    pr = period_returns.astype(float)
    if len(pr) < 2:
        return 0.0
    mu = float(pr.mean())
    sig = float(pr.std(ddof=1))
    if sig <= 0:
        return 0.0
    return (mu / sig) * math.sqrt(float(periods_per_year))


def summarize_performance(
    period_returns: pd.Series,
    periods_per_year: float,
    turnover: pd.Series | None = None,
) -> Dict[str, float]:
    """Compute baseline performance summary.

    Returns keys:
      - periods
      - periods_per_year
      - mean_period_return
      - ann_vol
      - sharpe
      - max_drawdown
      - avg_turnover (if turnover provided)
    """
    pr = period_returns.astype(float)
    eq = equity_curve(pr)

    out: Dict[str, float] = {
        "periods": float(len(pr)),
        "periods_per_year": float(periods_per_year),
        "mean_period_return": float(pr.mean()) if len(pr) else 0.0,
        "ann_vol": float(annualized_vol(pr, periods_per_year)),
        "sharpe": float(sharpe_ratio(pr, periods_per_year)),
        "max_drawdown": float(max_drawdown(eq)) if len(pr) else 0.0,
    }

    if turnover is not None and len(turnover):
        out["avg_turnover"] = float(turnover.astype(float).mean())

    return out