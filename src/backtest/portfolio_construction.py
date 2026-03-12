"""Portfolio construction (Phase 4 baseline).

This module contains reusable, strategy-agnostic portfolio construction steps:
- Cross-sectional selection (long top quantile, short bottom quantile)
- Equal-dollar long/short weights

No constraints, no risk model, no costs.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def rank_to_long_short(x: pd.Series, long_q: float, short_q: float) -> Tuple[pd.Index, pd.Index]:
    """Select long/short IDs using quantile thresholds.

    Parameters
    ----------
    x : pd.Series
        Signal values for a single date, indexed by asset id.
    long_q : float
        Upper quantile threshold (e.g., 0.9 for top decile).
    short_q : float
        Lower quantile threshold (e.g., 0.1 for bottom decile).

    Returns
    -------
    (pd.Index, pd.Index)
        (long_ids, short_ids)

    Notes
    -----
    Thresholds are inclusive (>=, <=), so ties can make the selected sets slightly larger.
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"rank_to_long_short expected a 1D signal Series, got DataFrame with shape {x.shape}"
            )
        x = x.iloc[:, 0]

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    x = x.dropna()
    if x.empty:
        return pd.Index([]), pd.Index([])

    q_hi = x.quantile(long_q)
    q_lo = x.quantile(short_q)

    long_mask = (x >= q_hi).to_numpy()
    short_mask = (x <= q_lo).to_numpy()

    long_ids = x.index[long_mask]
    short_ids = x.index[short_mask]

    return long_ids, short_ids


def make_equal_ls_weights(long_ids: pd.Index, short_ids: pd.Index) -> pd.Series:
    """Equal-dollar long/short weights.

    Long leg sums to +1, short leg sums to -1.

    Returns
    -------
    pd.Series
        Weights indexed by asset id.
    """
    nL = len(long_ids)
    nS = len(short_ids)
    if nL == 0 or nS == 0:
        return pd.Series(dtype=float)

    w = pd.Series(0.0, index=long_ids.union(short_ids))
    w.loc[long_ids] = 1.0 / nL
    w.loc[short_ids] = -1.0 / nS

    return w