

"""Volatility-Adjusted Returns (Volatility_AR) — price-based.

Phase 6 signal.

Definition (20-day risk-adjusted cumulative return):

For stock i on date t:
  R20_{i,t-1} = prod_{k=1..20} (1 + r_{i,t-k}) - 1
  sigma20_{i,t-1} = std(r_{i,t-k}, k=1..20)
  Volatility_AR_{i,t} = R20_{i,t-1} / sigma20_{i,t-1}

Timing:
- Signal on date t uses returns through t-1 (no lookahead).

Input requirements:
- DataFrame with columns: ['permno', 'date']
- Must contain either 'ret' or 'ret_total' for return calculation

Output:
- DataFrame with columns: ['permno', 'date', 'Volatility_AR']

Notes:
- Missing returns inside the 20-day window yield NaN (conservative).
- Division by 0 or near-0 volatility yields NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_NAME: Literal["Volatility_AR"] = "Volatility_AR"


@dataclass(frozen=True)
class VolAdjReturnConfig:
    window: int = 20
    vol_epsilon: float = 1e-12


def compute(dsf: pd.DataFrame, cfg: VolAdjReturnConfig | None = None) -> pd.DataFrame:
    """Compute Volatility-Adjusted Returns signal.

    Parameters
    ----------
    dsf : pd.DataFrame
        Must contain columns: permno, date, and either ret or ret_total.
    cfg : VolAdjReturnConfig | None
        Configuration (default window=20).

    Returns
    -------
    pd.DataFrame
        Columns: permno, date, Volatility_AR
    """
    if cfg is None:
        cfg = VolAdjReturnConfig()

    required = {"permno", "date"}
    missing = required - set(dsf.columns)
    if missing:
        raise ValueError(
            f"{SIGNAL_NAME} requires columns {sorted(required)}; missing {sorted(missing)}"
        )

    # Detect return column dynamically
    if "ret" in dsf.columns:
        ret_col = "ret"
    elif "ret_total" in dsf.columns:
        ret_col = "ret_total"
    else:
        raise ValueError(f"{SIGNAL_NAME} requires either 'ret' or 'ret_total' column.")

    out = dsf[["permno", "date", ret_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad = out[out["date"].isna()].head(5)
        raise ValueError(
            f"{SIGNAL_NAME}: could not parse some 'date' values to datetime. "
            f"Examples:\n{bad.to_string(index=False)}"
        )

    # Stable rolling behavior
    out = out.sort_values(["permno", "date"], kind="mergesort")

    r = out[ret_col].astype(float)

    # 20-day cumulative return ending at t-1
    one_plus = 1.0 + r

    def _cumprod_minus_one(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        return float(np.prod(x) - 1.0)

    cumret_end_t = (
        one_plus.groupby(out["permno"], sort=False)
        .rolling(window=cfg.window, min_periods=cfg.window)
        .apply(_cumprod_minus_one, raw=True)
        .reset_index(level=0, drop=True)
    )

    # 20-day realized volatility ending at t-1 (std of daily returns)
    def _std_nan_if_any(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        # ddof=0 is fine here; cross-sectional ranking is invariant to scaling
        return float(np.std(x, ddof=0))

    vol_end_t = (
        r.groupby(out["permno"], sort=False)
        .rolling(window=cfg.window, min_periods=cfg.window)
        .apply(_std_nan_if_any, raw=True)
        .reset_index(level=0, drop=True)
    )

    # Shift both so signal at t uses window ending at t-1
    cumret_end_t_minus_1 = cumret_end_t.groupby(out["permno"], sort=False).shift(1)
    vol_end_t_minus_1 = vol_end_t.groupby(out["permno"], sort=False).shift(1)

    denom = vol_end_t_minus_1
    denom = denom.where(denom.abs() > cfg.vol_epsilon)

    out[SIGNAL_NAME] = cumret_end_t_minus_1 / denom

    return out[["permno", "date", SIGNAL_NAME]]


# Backwards-compatible alias for the dynamic runner.
compute_signal = compute