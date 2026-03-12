

"""Turnover computations (Phase 4 baseline).

For the baseline backtest, turnover is defined as:
  turnover_t = sum_i |w_i,t - w_i,t-1|

where weights are portfolio weights at rebalance times and missing assets are treated as 0.
"""

from __future__ import annotations

import pandas as pd


def simple_turnover(w_prev: pd.Series, w_now: pd.Series) -> float:
    """Compute simple turnover = sum_i |w_now - w_prev|.

    Parameters
    ----------
    w_prev : pd.Series
        Previous rebalance weights indexed by asset id.
    w_now : pd.Series
        Current rebalance weights indexed by asset id.

    Returns
    -------
    float
        Sum absolute weight changes with missing treated as 0.
    """
    idx = w_prev.index.union(w_now.index)
    w_prev2 = w_prev.reindex(idx).fillna(0.0)
    w_now2 = w_now.reindex(idx).fillna(0.0)
    return float((w_now2 - w_prev2).abs().sum())