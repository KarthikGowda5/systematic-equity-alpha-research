

"""Idiosyncratic Volatility (idiosyncratic_vol) — risk-based.

Phase 6 signal (risk-based).

Goal
----
Measure a stock's idiosyncratic (market-unexplained) volatility using a rolling
market model, then form a low-idiosyncratic-volatility signal.

Definition
----------
1) Market return proxy r_m,t: equal-weight average return across all names each day.

2) For each stock i, estimate rolling market-model parameters over a 252-trading-day
   window ending at t-1 (no lookahead):

   r_{i,\tau} = alpha_{i,t} + beta_{i,t} * r_{m,\tau} + eps_{i,\tau},  \tau in [t-252, ..., t-1]

   Using rolling moments:
     beta = Cov(r_i, r_m) / Var(r_m)
     alpha = mean(r_i) - beta * mean(r_m)

3) Compute residual return at date t using parameters estimated through t-1:

   resid_{i,t} = r_{i,t} - (alpha_{i,t} + beta_{i,t} * r_{m,t})

4) Idiosyncratic volatility at date t is the rolling std of residual returns over
   the past 252 days ending at t-1:

   idio_vol_{i,t} = std(resid_{i, t-k}, k=1..252)

Signal direction
----------------
We use the common "idiosyncratic volatility puzzle" direction (high idio vol -> lower returns):

  idiosyncratic_vol_{i,t} = - idio_vol_{i,t}

So higher signal means lower idiosyncratic volatility -> long top decile.

Timing
------
- All rolling estimates and the idio vol window end at t-1 (via shift(1)).

Inputs
------
- DataFrame with columns: ['permno', 'date']
- Must contain either 'ret' or 'ret_total' as daily returns

Output
------
- DataFrame with columns: ['permno', 'date', 'idiosyncratic_vol']

Notes
-----
- Conservative NaN policy: if any NaNs in a required rolling window, output is NaN.
- If market variance is ~0, beta is NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SIGNAL_NAME: Literal["idiosyncratic_vol"] = "idiosyncratic_vol"


@dataclass(frozen=True)
class IdioVolConfig:
    window: int = 252
    var_epsilon: float = 1e-12


def compute(dsf: pd.DataFrame, cfg: IdioVolConfig | None = None) -> pd.DataFrame:
    """Compute low-idiosyncratic-volatility signal (-idio_vol) with a 252d window."""
    if cfg is None:
        cfg = IdioVolConfig()

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

    # Rolling market moments on the per-date market series
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
    alpha_end_t = ri_mean_end_t - beta_end_t * d["mkt_mean_end_t"].astype(float)

    # Use parameters estimated through t-1
    beta_t = beta_end_t.groupby(d["permno"], sort=False).shift(1)
    alpha_t = alpha_end_t.groupby(d["permno"], sort=False).shift(1)

    resid_t = r - (alpha_t + beta_t * d["mkt_ret"].astype(float))

    # Rolling std of residuals over window, ending at t-1
    def _std_nan_if_any(x: np.ndarray) -> float:
        if np.isnan(x).any():
            return np.nan
        return float(np.std(x, ddof=0))

    idio_vol_end_t = (
        resid_t.groupby(d["permno"], sort=False)
        .rolling(cfg.window, min_periods=cfg.window)
        .apply(_std_nan_if_any, raw=True)
        .reset_index(level=0, drop=True)
    )

    idio_vol_t = idio_vol_end_t.groupby(d["permno"], sort=False).shift(1)

    # Low-idio-vol signal
    d[SIGNAL_NAME] = -idio_vol_t

    return d[["permno", "date", SIGNAL_NAME]]


# Dynamic-runner compatibility
compute_idiosyncratic_vol = compute