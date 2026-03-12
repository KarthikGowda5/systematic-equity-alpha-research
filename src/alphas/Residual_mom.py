

"""Residual Momentum (Residual_mom) — price-based.

Phase 6 signal.

Goal
----
Compute momentum on idiosyncratic (market-residualized) daily returns.

Definition
----------
1) Market return (proxy): equal-weight average return across all names each day.

2) For each stock i, estimate a rolling market model with intercept over a 252-trading-day window
   ending at t-1 (to avoid lookahead):

   r_{i,\tau} = alpha_{i,t} + beta_{i,t} * r_{m,\tau} + eps_{i,\tau},  for \tau in [t-252, ..., t-1]

   Using rolling moments:
     beta = cov(r_i, r_m) / var(r_m)
     alpha = mean(r_i) - beta * mean(r_m)

3) Compute daily residual return at date t using parameters estimated through t-1:

   resid_{i,t} = r_{i,t} - (alpha_{i,t} + beta_{i,t} * r_{m,t})

4) Residual momentum (12-1 style) at date t is the sum of residual returns over the prior 252 days,
   skipping the most recent 21 days, and ending at t-1:

   Residual_mom_{i,t} = sum_{k=21..252} resid_{i,t-k}

Timing
------
- Signal on date t uses only information through t-1.

Inputs
------
- DataFrame with columns: ['permno', 'date']
- Must contain either 'ret' or 'ret_total' as the daily return column.

Output
------
- DataFrame with columns: ['permno', 'date', 'Residual_mom']

Notes
-----
- This implementation uses an equal-weight market proxy from the same panel.
- If required rolling windows contain NaNs, the signal is NaN (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_NAME: Literal["Residual_mom"] = "Residual_mom"


@dataclass(frozen=True)
class ResidualMomConfig:
    reg_window: int = 252
    mom_window: int = 252
    skip: int = 21
    var_epsilon: float = 1e-12


def compute(dsf: pd.DataFrame, cfg: ResidualMomConfig | None = None) -> pd.DataFrame:
    """Compute residual momentum signal."""
    if cfg is None:
        cfg = ResidualMomConfig()

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

    d = dsf[["permno", "date", ret_col]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().any():
        bad = d[d["date"].isna()].head(5)
        raise ValueError(
            f"{SIGNAL_NAME}: could not parse some 'date' values to datetime. Examples:\n"
            f"{bad.to_string(index=False)}"
        )

    d = d.sort_values(["permno", "date"], kind="mergesort")
    r = d[ret_col].astype(float)

    # Equal-weight market proxy per date
    mkt = d.groupby("date", sort=False)[ret_col].mean().astype(float)
    # Align market return back to rows
    d["mkt_ret"] = d["date"].map(mkt)

    # Rolling moments for market (same for all names)
    mkt_ret = d["mkt_ret"].astype(float)
    mkt_mean_end_t = mkt_ret.groupby(d["permno"], sort=False).transform(lambda x: x)  # placeholder to keep shape
    # Compute market rolling stats once on the per-date series, then map back
    mkt_mean = mkt.rolling(cfg.reg_window, min_periods=cfg.reg_window).mean()
    mkt_mean2 = (mkt * mkt).rolling(cfg.reg_window, min_periods=cfg.reg_window).mean()
    var_mkt = mkt_mean2 - mkt_mean * mkt_mean

    # Map rolling market stats to each row by date
    d["mkt_mean_end_t"] = d["date"].map(mkt_mean)
    d["var_mkt_end_t"] = d["date"].map(var_mkt)

    # Rolling stats for each stock
    # mean(r_i)
    ri_mean_end_t = (
        r.groupby(d["permno"], sort=False)
        .rolling(cfg.reg_window, min_periods=cfg.reg_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # mean(r_i * r_m)
    rim = (r * d["mkt_ret"].astype(float))
    rim_mean_end_t = (
        rim.groupby(d["permno"], sort=False)
        .rolling(cfg.reg_window, min_periods=cfg.reg_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # cov = E[r_i r_m] - E[r_i]E[r_m]
    cov_end_t = rim_mean_end_t - ri_mean_end_t * d["mkt_mean_end_t"].astype(float)

    # beta/alpha estimated on window ending at t-1 (shift by 1 day)
    var_end_t = d["var_mkt_end_t"].astype(float)
    var_end_t = var_end_t.where(var_end_t.abs() > cfg.var_epsilon)

    beta_end_t = cov_end_t / var_end_t
    alpha_end_t = ri_mean_end_t - beta_end_t * d["mkt_mean_end_t"].astype(float)

    beta_t = beta_end_t.groupby(d["permno"], sort=False).shift(1)
    alpha_t = alpha_end_t.groupby(d["permno"], sort=False).shift(1)

    # Residual return at t using params through t-1
    resid_t = r - (alpha_t + beta_t * d["mkt_ret"].astype(float))

    # Residual momentum: sum of residual returns over [t-252 .. t-22] (skip last 21), ending t-1.
    # Implement as rolling sum over mom_window, shift by 1 to end at t-1, then subtract the most recent `skip` days.
    resid_roll_sum_end_t = (
        resid_t.groupby(d["permno"], sort=False)
        .rolling(cfg.mom_window, min_periods=cfg.mom_window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    resid_roll_sum_end_t_minus_1 = resid_roll_sum_end_t.groupby(d["permno"], sort=False).shift(1)

    # Sum of most recent `skip` residual days ending at t-1
    resid_skip_sum_end_t = (
        resid_t.groupby(d["permno"], sort=False)
        .rolling(cfg.skip, min_periods=cfg.skip)
        .sum()
        .reset_index(level=0, drop=True)
    )
    resid_skip_sum_end_t_minus_1 = resid_skip_sum_end_t.groupby(d["permno"], sort=False).shift(1)

    d[SIGNAL_NAME] = resid_roll_sum_end_t_minus_1 - resid_skip_sum_end_t_minus_1

    return d[["permno", "date", SIGNAL_NAME]]


# Dynamic-runner compatibility
compute_residual_mom = compute