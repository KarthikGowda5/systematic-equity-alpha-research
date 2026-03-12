"""Short-Term Reversal (STR) — price-based.

Phase 6 signal #1.

Definition (5-day cumulative return, contrarian):

For stock i on date t:
  R5_{i,t-1} = prod_{k=1..5} (1 + ret_{i,t-k}) - 1
  STR_{i,t}  = - R5_{i,t-1}

Timing:
- Signal on date t uses returns through t-1 (no lookahead).

Input requirements:
- DataFrame with columns: ['permno', 'date']
- Must contain either 'ret' or 'ret_total' for return calculation

Output:
- DataFrame with columns: ['permno', 'date', 'STR']

Notes:
- Missing returns inside the 5-day window will yield NaN (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_NAME: Literal["STR"] = "STR"


@dataclass(frozen=True)
class STRConfig:
    window: int = 5


def compute(dsf: pd.DataFrame, cfg: STRConfig | None = None) -> pd.DataFrame:
    """Compute Short-Term Reversal (STR) signal.

    Parameters
    ----------
    dsf : pd.DataFrame
        Must contain columns: permno, date, ret.
    cfg : STRConfig | None
        Configuration (default window=5).

    Returns
    -------
    pd.DataFrame
        Columns: permno, date, STR
    """
    if cfg is None:
        cfg = STRConfig()

    required = {"permno", "date"}
    missing = required - set(dsf.columns)
    if missing:
        raise ValueError(f"STR signal requires columns {sorted(required)}; missing {sorted(missing)}")

    # Detect return column dynamically
    if "ret" in dsf.columns:
        ret_col = "ret"
    elif "ret_total" in dsf.columns:
        ret_col = "ret_total"
    else:
        raise ValueError("STR requires either 'ret' or 'ret_total' column.")

    out = dsf[["permno", "date", ret_col]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad = out[out["date"].isna()].head(5)
        raise ValueError(
            "STR signal: could not parse some 'date' values to datetime. "
            f"Examples:\n{bad.to_string(index=False)}"
        )

    # Ensure stable rolling behavior
    out = out.sort_values(["permno", "date"], kind="mergesort")

    # 5-day cumulative return ending at t-1.
    # We compute cumret5_end_t = prod_{k=0..window-1}(1+ret_{t-k}) - 1,
    # then shift by 1 to end at t-1.
    one_plus = 1.0 + out[ret_col].astype(float)

    def _cumprod_minus_one(x: np.ndarray) -> float:
        # If any NaN in window -> NaN
        if np.isnan(x).any():
            return np.nan
        return float(np.prod(x) - 1.0)

    cumret_end_t = (
        one_plus.groupby(out["permno"], sort=False)
        .rolling(window=cfg.window, min_periods=cfg.window)
        .apply(_cumprod_minus_one, raw=True)
        .reset_index(level=0, drop=True)
    )

    cumret_end_t_minus_1 = cumret_end_t.groupby(out["permno"], sort=False).shift(1)

    out[SIGNAL_NAME] = -cumret_end_t_minus_1

    return out[["permno", "date", SIGNAL_NAME]]


# Backwards-compatible alias if the engine expects a specific function name.
compute_STR = compute