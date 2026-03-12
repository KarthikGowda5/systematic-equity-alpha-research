

"""Beta (beta) — risk-based.

Phase 6 signal (risk-based), starting with beta.

Definition
----------
We estimate a rolling market beta for each stock i using a 252-trading-day window
ending at t-1 (no lookahead), with an equal-weight market proxy built from the
same panel:

  beta_{i,t} = Cov(r_i, r_m) / Var(r_m)

Signal direction
----------------
We implement the low-beta signal (Betting-Against-Beta style):

  beta_signal_{i,t} = - beta_{i,t}

So higher signal means lower beta -> long top decile, short bottom decile.

Timing
------
- beta is computed on a rolling window ending at t-1 (via shift(1)).

Inputs
------
- DataFrame with columns: ['permno', 'date']
- Must contain either 'ret' or 'ret_total' as daily returns

Output
------
- DataFrame with columns: ['permno', 'date', 'beta']

Notes
-----
- Equal-weight market return is computed per date as the average return across all names.
- If any NaNs appear within the rolling window, beta is NaN (conservative).
- If market variance is ~0, beta is NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_NAME: Literal["beta"] = "beta"


@dataclass(frozen=True)
class BetaConfig:
    window: int = 252
    var_epsilon: float = 1e-12


def compute(dsf: pd.DataFrame, cfg: BetaConfig | None = None) -> pd.DataFrame:
    """Compute low-beta signal (-beta) with a 252d rolling window."""
    if cfg is None:
        cfg = BetaConfig()

    required = {"permno", "date"}
    missing = required - set(dsf.columns)
    if missing:
        raise ValueError(f"{SIGNAL_NAME} requires columns {sorted(required)}; missing {sorted(missing)}")

    # Detect return column dynamically
    if "ret" in dsf.columns:
        ret_col = "ret"
    elif "ret_total" in dsf.columns:
        ret_col = "ret_total"
    else:
        raise ValueError(f"{SIGNAL_NAME} requires either 'ret' or 'ret_total' column.")

    d = dsf[["permno", "date", ret_col]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().any():
        bad = d[d["date"].isna()].head(5)
        raise ValueError(
            f"{SIGNAL_NAME}: could not parse some 'date' values to datetime. Examples:\n{bad.to_string(index=False)}"
        )

    d = d.sort_values(["permno", "date"], kind="mergesort")
    r = d[ret_col].astype(float)

    # Equal-weight market proxy per date
    mkt = d.groupby("date", sort=False)[ret_col].mean().astype(float)
    d["mkt_ret"] = d["date"].map(mkt)

    # Rolling market moments computed on the per-date market series
    mkt_mean = mkt.rolling(cfg.window, min_periods=cfg.window).mean()
    mkt_mean2 = (mkt * mkt).rolling(cfg.window, min_periods=cfg.window).mean()
    var_mkt = mkt_mean2 - mkt_mean * mkt_mean

    # Map rolling market stats to each row by date
    d["mkt_mean_end_t"] = d["date"].map(mkt_mean)
    d["var_mkt_end_t"] = d["date"].map(var_mkt)

    # Rolling stock moments per permno
    ri_mean_end_t = (
        r.groupby(d["permno"], sort=False)
        .rolling(cfg.window, min_periods=cfg.window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rim = r * d["mkt_ret"].astype(float)
    rim_mean_end_t = (
        rim.groupby(d["permno"], sort=False)
        .rolling(cfg.window, min_periods=cfg.window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cov_end_t = rim_mean_end_t - ri_mean_end_t * d["mkt_mean_end_t"].astype(float)

    var_end_t = d["var_mkt_end_t"].astype(float)
    var_end_t = var_end_t.where(var_end_t.abs() > cfg.var_epsilon)

    beta_end_t = cov_end_t / var_end_t

    # Use window ending at t-1
    beta_t = beta_end_t.groupby(d["permno"], sort=False).shift(1)

    # Low-beta signal (negative beta)
    d[SIGNAL_NAME] = -beta_t

    return d[["permno", "date", SIGNAL_NAME]]


# Dynamic-runner compatibility
compute_beta  = compute