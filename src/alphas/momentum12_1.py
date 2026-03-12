"""
12–1 Momentum Signal (Price-Based)

Definition (daily data):
- Lookback window: 252 trading days (~12 months)
- Skip window: 21 trading days (~1 month)
- Signal at date t uses returns from t-252 to t-21

Output:
- DataFrame with columns: permno, date, mom_12_1

Assumptions:
- Input panel contains: permno, date, ret_total
- ret_total is total return in decimal form (e.g., 0.01 = 1%)
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_momentum_12_1(
    df: pd.DataFrame,
    id_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret_total",
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    Compute 12–1 cross-sectional momentum signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain id_col, date_col, ret_col.
    lookback : int
        Number of trading days in total lookback window (~252).
    skip : int
        Number of most recent trading days to skip (~21).

    Returns
    -------
    pd.DataFrame
        Columns: [permno, date, mom_12_1]
    """

    required = {id_col, date_col, ret_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for momentum: {sorted(missing)}")

    d = df[[id_col, date_col, ret_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col])

    # Convert to log returns for numerical stability
    # Guard against ret_total == -1.0 (delistings), which would yield -inf in log1p.
    r = d[ret_col].astype(float)
    r = np.clip(r, -0.999999, None)
    d["log_ret"] = np.log1p(r)

    # Rolling cumulative log return over lookback window
    # Then subtract the most recent 'skip' window
    def _mom(group: pd.DataFrame) -> pd.Series:
        lr = group["log_ret"]
        # rolling sum of full lookback window
        full = lr.rolling(window=lookback, min_periods=lookback).sum()
        # rolling sum of last 'skip' days
        recent = lr.rolling(window=skip, min_periods=skip).sum()
        # shift recent so it aligns with end of lookback window
        recent = recent.shift(lookback - skip)
        return full - recent

    d["mom_12_1"] = (
        d.groupby(id_col, sort=False)
        .apply(_mom)
        .reset_index(level=0, drop=True)
    )

    return d[[id_col, date_col, "mom_12_1"]]


# ------------------------------------------------------------------
# Dynamic loader compatibility (used by run_baseline)
# ------------------------------------------------------------------

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper so the dynamic runner can load this alpha.

    `run_baseline.py` expects the returned DataFrame to contain columns:
      [permno, date, <module_name>]

    For this module, the expected name is: 'momentum12_1'.
    """
    sig = compute_momentum_12_1(df)
    # Baseline expects the signal column name to match the module/signal string.
    return sig.rename(columns={"mom_12_1": "momentum12_1"})