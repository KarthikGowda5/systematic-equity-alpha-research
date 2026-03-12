

"""Alpha validation framework (Phase 5).

This module provides a reusable, single-signal validation suite:
- Cross-sectional rank IC series
- IC mean + t-stat (Newey-West HAC)
- Decay curve (IC at horizons 1..H)
- Rolling IC stability (e.g., 252 trading days)
- Subperiod analysis (user-provided splits)
- High-vol vs low-vol regime split (single rule; user-provided)

Inputs
------
A panel DataFrame with at least:
- permno (id)
- date
- ret_total (daily total return)
- <signal_col> (alpha signal)

Optionally, a universe mask DataFrame with:
- date, permno, in_universe (bool)

Design
------
- All ICs are computed cross-sectionally per date.
- Rank IC uses Spearman correlation (implemented via ranks + Pearson).
- Forward returns are computed as cumulative log returns over horizon h.

Notes
-----
- This module is deliberately “single-signal” and stat-focused.
- It does not do portfolio construction; that remains in src/backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeSpec:
    """Definition of a regime split.

    rule must return a Series indexed by date with boolean values:
    True  -> high-vol (or regime A)
    False -> low-vol  (or regime B)
    """

    name: str
    rule: Callable[[pd.DataFrame, str, str, str], pd.Series]


@dataclass(frozen=True)
class ValidationSpec:
    horizons: Tuple[int, ...] = tuple(range(1, 21))
    rolling_window: int = 252
    nw_lags: int = 5


# ----------------------------
# Helpers
# ----------------------------

def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation via ranks + Pearson.

    Returns np.nan if insufficient data.
    """
    if x.size < 3 or y.size < 3:
        return np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    # Pandas corr handles constant series -> nan
    return float(xr.corr(yr))


def newey_west_tstat(x: pd.Series, lags: int) -> Tuple[float, float, float]:
    """Compute mean, HAC std error, and t-stat for a series.

    Parameters
    ----------
    x : pd.Series
        Time series of observations (e.g., daily/weekly IC).
    lags : int
        Newey-West maximum lag.

    Returns
    -------
    mean, se_hac, t
    """
    z = x.dropna().astype(float).values
    n = z.shape[0]
    if n < 3:
        return (float(np.nan), float(np.nan), float(np.nan))

    mu = float(z.mean())
    u = z - mu

    # Gamma_0
    gamma0 = float(np.dot(u, u) / n)

    var = gamma0
    q = min(int(lags), n - 1)
    for k in range(1, q + 1):
        w = 1.0 - (k / (q + 1.0))
        gam = float(np.dot(u[k:], u[:-k]) / n)
        var += 2.0 * w * gam

    # HAC standard error of the mean
    se = float(np.sqrt(max(var, 0.0) / n))
    t = float(mu / se) if se > 0 else float(np.nan)
    return (mu, se, t)


def compute_forward_log_return(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    ret_col: str,
    horizon: int,
) -> pd.Series:
    """Compute per-(date,id) forward cumulative log return over horizon.

    For each asset i and date t, compute sum_{j=1..h} log(1+r_{t+j}).

    Returns
    -------
    pd.Series
        Indexed by (date, id). The value is forward log return.
    """
    _require_cols(df, [id_col, date_col, ret_col])

    d = df[[id_col, date_col, ret_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col])
    d["log_ret"] = np.log1p(d[ret_col].astype(float))

    # For each asset, forward sum of next h log returns.
    def _fwd_sum(g: pd.DataFrame) -> pd.Series:
        lr = g["log_ret"]
        # rolling sum over horizon, then shift so it is aligned to start date t
        out = lr.rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
        return out

    fwd = (
        d.groupby(id_col, sort=False)
        .apply(_fwd_sum)
        .reset_index(level=0, drop=True)
    )

    fwd.index = pd.MultiIndex.from_frame(d[[date_col, id_col]])
    return fwd


def apply_universe_mask_to_panel(
    panel: pd.DataFrame,
    universe_mask: Optional[pd.DataFrame],
    id_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Filter a panel by (date,id)->in_universe mask if provided."""
    if universe_mask is None:
        return panel

    _require_cols(universe_mask, [date_col, id_col, "in_universe"])

    p = panel.copy()
    p[date_col] = pd.to_datetime(p[date_col])

    m = universe_mask[[date_col, id_col, "in_universe"]].copy()
    m[date_col] = pd.to_datetime(m[date_col])
    m["in_universe"] = m["in_universe"].astype("boolean").fillna(False).astype(bool)
    m = m.drop_duplicates(subset=[date_col, id_col])

    out = p.merge(m, on=[date_col, id_col], how="left")
    out["in_universe"] = out["in_universe"].astype("boolean").fillna(False).astype(bool)
    return out[out["in_universe"]].drop(columns=["in_universe"])


# ----------------------------
# Core computations
# ----------------------------

def compute_rank_ic_series(
    panel: pd.DataFrame,
    signal_col: str,
    fwd_ret: pd.Series,
    id_col: str = "permno",
    date_col: str = "date",
) -> pd.Series:
    """Compute cross-sectional rank IC_t between signal(t) and fwd_ret(t)."""
    _require_cols(panel, [id_col, date_col, signal_col])

    d = panel[[id_col, date_col, signal_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])

    sig = d.set_index([date_col, id_col])[signal_col].astype(float)

    # Align indices and drop missing
    joined = pd.concat([sig.rename("signal"), fwd_ret.rename("fwd")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    # Compute per-date IC
    def _ic(g: pd.DataFrame) -> float:
        return _spearman_corr(g["signal"], g["fwd"])

    ic = joined.reset_index().groupby(date_col).apply(_ic)
    ic.name = "rank_ic"
    return ic.sort_index()


def compute_decay_curve(
    panel: pd.DataFrame,
    signal_col: str,
    spec: ValidationSpec,
    universe_mask: Optional[pd.DataFrame] = None,
    id_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret_total",
) -> pd.DataFrame:
    """Compute mean IC and t-stat by horizon."""

    p = apply_universe_mask_to_panel(panel, universe_mask, id_col=id_col, date_col=date_col)

    rows = []
    for h in spec.horizons:
        fwd = compute_forward_log_return(p, id_col=id_col, date_col=date_col, ret_col=ret_col, horizon=h)
        ic = compute_rank_ic_series(p, signal_col=signal_col, fwd_ret=fwd, id_col=id_col, date_col=date_col)
        mean_ic, se, t = newey_west_tstat(ic, lags=spec.nw_lags)
        rows.append({"horizon": h, "mean_rank_ic": mean_ic, "nw_se": se, "nw_t": t, "n_obs": int(ic.dropna().shape[0])})

    return pd.DataFrame(rows)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def subperiod_stats(
    ic: pd.Series,
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
    nw_lags: int,
) -> pd.DataFrame:
    """Compute mean/t-stat for IC series across user-provided splits."""
    rows = []
    for start, end, label in splits:
        s = ic.loc[(ic.index >= start) & (ic.index <= end)]
        mean_ic, se, t = newey_west_tstat(s, lags=nw_lags)
        rows.append({"label": label, "start": start, "end": end, "mean_rank_ic": mean_ic, "nw_se": se, "nw_t": t, "n_obs": int(s.dropna().shape[0])})
    return pd.DataFrame(rows)


def run_validation(
    panel: pd.DataFrame,
    signal_col: str,
    spec: ValidationSpec = ValidationSpec(),
    universe_mask: Optional[pd.DataFrame] = None,
    id_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret_total",
    subperiod_splits: Optional[List[Tuple[str, str, str]]] = None,
    regime_spec: Optional[RegimeSpec] = None,
) -> Dict[str, object]:
    """Run the full validation suite for a single signal.

    Parameters
    ----------
    subperiod_splits:
        Optional list of (start_date_str, end_date_str, label).
    regime_spec:
        Optional regime definition. rule(panel, id_col, date_col, ret_col) -> Series[bool] indexed by date.

    Returns
    -------
    dict with keys:
        - ic_series (pd.Series)
        - ic_summary (dict)
        - decay (pd.DataFrame)
        - rolling_ic (pd.Series)
        - subperiods (pd.DataFrame | None)
        - regimes (dict | None)
    """
    _require_cols(panel, [id_col, date_col, ret_col, signal_col])

    # Apply universe mask to the panel used for all validations.
    p = apply_universe_mask_to_panel(panel, universe_mask, id_col=id_col, date_col=date_col)

    # Horizon-1 IC series is the main time series object.
    fwd1 = compute_forward_log_return(p, id_col=id_col, date_col=date_col, ret_col=ret_col, horizon=1)
    ic = compute_rank_ic_series(p, signal_col=signal_col, fwd_ret=fwd1, id_col=id_col, date_col=date_col)

    mean_ic, se, t = newey_west_tstat(ic, lags=spec.nw_lags)
    ic_summary = {
        "mean_rank_ic": mean_ic,
        "nw_se": se,
        "nw_t": t,
        "n_obs": int(ic.dropna().shape[0]),
        "nw_lags": spec.nw_lags,
    }

    decay = compute_decay_curve(
        panel=p,
        signal_col=signal_col,
        spec=spec,
        universe_mask=None,  # already applied
        id_col=id_col,
        date_col=date_col,
        ret_col=ret_col,
    )

    roll = rolling_mean(ic, window=spec.rolling_window)
    roll.name = f"rolling_mean_rank_ic_{spec.rolling_window}"

    sub_df = None
    if subperiod_splits:
        splits_parsed = []
        for s, e, label in subperiod_splits:
            splits_parsed.append((pd.to_datetime(s), pd.to_datetime(e), label))
        sub_df = subperiod_stats(ic, splits_parsed, nw_lags=spec.nw_lags)

    regimes_out = None
    if regime_spec is not None:
        reg = regime_spec.rule(p, id_col, date_col, ret_col)
        reg = reg.dropna().astype(bool)
        # Align to IC dates
        reg = reg.reindex(ic.index).dropna().astype(bool)
        ic_hi = ic[reg]
        ic_lo = ic[~reg]
        hi_mean, hi_se, hi_t = newey_west_tstat(ic_hi, lags=spec.nw_lags)
        lo_mean, lo_se, lo_t = newey_west_tstat(ic_lo, lags=spec.nw_lags)
        regimes_out = {
            regime_spec.name: {
                "high": {"mean_rank_ic": hi_mean, "nw_se": hi_se, "nw_t": hi_t, "n_obs": int(ic_hi.dropna().shape[0])},
                "low": {"mean_rank_ic": lo_mean, "nw_se": lo_se, "nw_t": lo_t, "n_obs": int(ic_lo.dropna().shape[0])},
            }
        }

    return {
        "ic_series": ic,
        "ic_summary": ic_summary,
        "decay": decay,
        "rolling_ic": roll,
        "subperiods": sub_df,
        "regimes": regimes_out,
    }


# ----------------------------
# A default high/low vol regime rule (realized market vol)
# ----------------------------

def realized_vol_regime_rule(
    panel: pd.DataFrame,
    id_col: str,
    date_col: str,
    ret_col: str,
    market_id: Optional[int] = None,
    window: int = 20,
    split: float = 0.5,
) -> pd.Series:
    """Compute a high-vol regime indicator from realized volatility.

    We compute a market proxy time series using either:
    - a specified `market_id` (permno) if provided
    - otherwise, equal-weight average of daily returns across available names
      (within the provided panel and any universe mask already applied)

    Then realized vol = rolling std of daily returns over `window`.
    High-vol is defined as realized vol >= rolling median (split=0.5) or
    the split quantile if split != 0.5.

    Returns
    -------
    pd.Series[bool] indexed by date
    """
    _require_cols(panel, [id_col, date_col, ret_col])

    d = panel[[id_col, date_col, ret_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])

    if market_id is not None:
        m = d[d[id_col] == market_id].copy()
        mkt = m.set_index(date_col)[ret_col].astype(float).sort_index()
    else:
        # equal-weight average return across names each day
        mkt = d.groupby(date_col)[ret_col].mean().astype(float).sort_index()

    rv = mkt.rolling(window=window, min_periods=window).std()

    if split == 0.5:
        thr = rv.rolling(window=window * 12, min_periods=window * 6).median()
    else:
        thr = rv.rolling(window=window * 12, min_periods=window * 6).quantile(split)

    reg = rv >= thr
    reg.name = "high_vol"
    return reg