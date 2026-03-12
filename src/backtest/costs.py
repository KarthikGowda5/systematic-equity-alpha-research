

"""Transaction cost & execution modeling utilities (Phase 9).

Design goals:
- Keep the backtest engine clean by centralizing cost model logic here.
- All functions are pure (no I/O) and rebalance-schedule agnostic.
- Use lagged inputs (t-1) when executing trades at t to avoid lookahead.

Implemented components:
1) Bid/ask quoted spread proxy (half-spread crossing)
2) Volatility-proportional slippage (k * sigma)
3) Participation constraint: max 10% of dollar ADV (clips trade size)
4) Turnover-driven cost (linear in turnover)

Notes:
- Costs are returned in *return units* (cost per $ traded) to be applied against portfolio returns.
- Participation constraint requires a portfolio notional (PORTFOLIO_VALUE) to translate weights -> dollars.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def resolve_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first column name that exists in df from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ------------------------
# 1) Spread proxy
# ------------------------

def compute_spread_proxy(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    bid_col: str,
    ask_col: str,
    bidlo_col: str,
    askhi_col: str,
    cap: float = 0.20,
) -> pd.DataFrame:
    """Compute quoted relative spread and 1-day lag per asset.

    Primary: BID/ASK (quoted spread).
    Fallback: BIDLO/ASKHI (extremes) only when BID/ASK are missing/invalid.

    Returns columns: [date_col, id_col, 'qspread', 'qspread_lag1'].
    """
    x = df[[id_col, date_col, bid_col, ask_col, bidlo_col, askhi_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col])

    for c in [bid_col, ask_col, bidlo_col, askhi_col]:
        x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)

    bid = x[bid_col]
    ask = x[ask_col]
    bidlo = x[bidlo_col]
    askhi = x[askhi_col]

    # Validity masks
    m_primary = (bid > 0) & (ask > 0) & (ask > bid)
    m_fallback = (~m_primary) & (bidlo > 0) & (askhi > 0) & (askhi > bidlo)

    use_bid = pd.Series(np.nan, index=x.index, dtype=float)
    use_ask = pd.Series(np.nan, index=x.index, dtype=float)

    use_bid.loc[m_primary] = bid.loc[m_primary]
    use_ask.loc[m_primary] = ask.loc[m_primary]

    use_bid.loc[m_fallback] = bidlo.loc[m_fallback]
    use_ask.loc[m_fallback] = askhi.loc[m_fallback]

    mid = (use_ask + use_bid) / 2.0
    qspread = (use_ask - use_bid) / mid

    # Hygiene: finite, non-negative, capped
    qspread = qspread.where(np.isfinite(qspread), np.nan)
    qspread = qspread.clip(lower=0.0, upper=float(cap))

    x["qspread"] = qspread

    x = x.sort_values([id_col, date_col])
    x["qspread_lag1"] = x.groupby(id_col)["qspread"].shift(1)

    return x[[date_col, id_col, "qspread", "qspread_lag1"]]


def spread_cost_from_dw(dw: pd.Series, qspread_lag1: pd.Series) -> float:
    """Half-spread crossing cost applied to traded notional (weights).

    Cost in return units:
        sum_i |dw_i| * 0.5 * qspread_lag1_i

    Missing qspread is treated as 0 cost (per project choice).
    """
    q = pd.to_numeric(qspread_lag1, errors="coerce").fillna(0.0)
    q = q.clip(lower=0.0)
    return float((dw.abs() * 0.5 * q).sum())


# ------------------------
# 2) Volatility-proportional slippage
# ------------------------

def compute_realized_vol_proxy(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    ret_col: str,
    lookback: int = 20,
    cap: float = 0.20,
) -> pd.DataFrame:
    """Compute rolling realized vol (std of daily returns) and 1-day lag per asset.

    Returns columns: [date_col, id_col, 'rv', 'rv_lag1'].

    Hygiene:
    - Coerce ret to float
    - Clip rv to [0, cap]
    """
    x = df[[id_col, date_col, ret_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col])
    x[ret_col] = pd.to_numeric(x[ret_col], errors="coerce").astype(float)

    x = x.sort_values([id_col, date_col])
    rv = x.groupby(id_col)[ret_col].rolling(int(lookback), min_periods=int(lookback)).std()
    rv = rv.reset_index(level=0, drop=True)

    rv = rv.where(np.isfinite(rv), np.nan)
    rv = rv.clip(lower=0.0, upper=float(cap))

    x["rv"] = rv
    x["rv_lag1"] = x.groupby(id_col)["rv"].shift(1)

    return x[[date_col, id_col, "rv", "rv_lag1"]]


def vol_slippage_cost_from_dw(dw: pd.Series, rv_lag1: pd.Series, k: float) -> float:
    """Volatility-proportional slippage cost: sum_i |dw_i| * (k * rv_lag1_i).

    Missing rv is treated as 0 cost.
    """
    rv = pd.to_numeric(rv_lag1, errors="coerce").fillna(0.0)
    rv = rv.clip(lower=0.0)
    return float((dw.abs() * float(k) * rv).sum())


# ------------------------
# 3) Participation constraint (max % of dollar ADV)
# ------------------------

def compute_dollar_adv_proxy(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    prc_col: str,
    vol_col: str,
    lookback: int = 20,
) -> pd.DataFrame:
    """Compute rolling dollar ADV = mean(|price| * volume) and 1-day lag per asset.

    Returns columns: [date_col, id_col, 'dadv', 'dadv_lag1'].

    Notes:
    - Uses abs(price) to handle CRSP signed prices.
    - Uses simple mean over the lookback window.
    """
    x = df[[id_col, date_col, prc_col, vol_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col])
    x[prc_col] = pd.to_numeric(x[prc_col], errors="coerce").astype(float)
    x[vol_col] = pd.to_numeric(x[vol_col], errors="coerce").astype(float)

    x = x.sort_values([id_col, date_col])
    dollar_vol = x[prc_col].abs() * x[vol_col]

    dadv = dollar_vol.groupby(x[id_col]).rolling(int(lookback), min_periods=int(lookback)).mean()
    dadv = dadv.reset_index(level=0, drop=True)

    dadv = dadv.where(np.isfinite(dadv), np.nan)

    x["dadv"] = dadv
    x["dadv_lag1"] = x.groupby(id_col)["dadv"].shift(1)

    return x[[date_col, id_col, "dadv", "dadv_lag1"]]


def apply_participation_constraint_to_weights(
    w_prev: pd.Series,
    w_target: pd.Series,
    dadv_lag1: pd.Series,
    portfolio_value: float,
    max_participation: float = 0.10,
) -> Tuple[pd.Series, pd.Series]:
    """Clip the trade (dw) implied by w_target so dollars traded <= max_participation * dollar ADV.

    Inputs:
    - w_prev: previous used weights
    - w_target: desired new weights
    - dadv_lag1: lagged dollar ADV per asset (aligned by index)
    - portfolio_value: notional capital in dollars

    Returns:
    - w_exec: executed weights after clipping
    - dw_exec: executed trade weights

    Missing dadv -> no clipping for that asset (treat as infinite capacity).
    """
    idx = w_prev.index.union(w_target.index)
    w0 = w_prev.reindex(idx).fillna(0.0)
    w1 = w_target.reindex(idx).fillna(0.0)

    dw = w1 - w0

    # Max dollar trade per asset
    cap_dollars = float(max_participation) * pd.to_numeric(dadv_lag1.reindex(idx), errors="coerce")
    # If missing, set to +inf so it doesn't clip
    cap_dollars = cap_dollars.fillna(np.inf)

    # Convert to max weight change
    if float(portfolio_value) <= 0:
        raise ValueError("portfolio_value must be > 0 for participation constraint")

    cap_w = cap_dollars / float(portfolio_value)

    # Clip each asset's trade weight
    dw_exec = dw.clip(lower=-cap_w, upper=cap_w)
    w_exec = w0 + dw_exec

    return w_exec, dw_exec


# ------------------------
# 4) Turnover-driven cost
# ------------------------

def turnover_cost_from_dw(dw: pd.Series, cost_per_dollar: float) -> float:
    """Linear turnover cost: cost_per_dollar * sum_i |dw_i|."""
    return float(float(cost_per_dollar) * dw.abs().sum())