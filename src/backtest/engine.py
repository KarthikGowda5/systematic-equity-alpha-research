"""Backtest engine plumbing (Phase 4).

- generating rebalance dates (D/W/BW/TW/M)
- building long/short quantile weights cross-sectionally
- computing holding-period portfolio returns between rebalance dates
- computing simple turnover = sum_i |w_t - w_prev|

It is intentionally "baseline" (no transaction costs by default, no risk model by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from src.backtest.metrics import equity_curve as compute_equity_curve
from src.backtest.metrics import summarize_performance
from src.backtest.portfolio_construction import make_equal_ls_weights, rank_to_long_short
from src.backtest.turnover import simple_turnover
from src.backtest.optimizer import optimize_mean_variance, OptimizerParams

# --- Phase 9: Transaction cost and execution constraint helpers ---
from src.backtest.costs import (
    resolve_col,
    compute_spread_proxy,
    spread_cost_from_dw,
    compute_realized_vol_proxy,
    vol_slippage_cost_from_dw,
    compute_dollar_adv_proxy,
    apply_participation_constraint_to_weights,
    turnover_cost_from_dw,
)


WEEKDAY_MAP = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}




def get_rebalance_dates(trading_dates: pd.DatetimeIndex, settings: Dict[str, Any]) -> pd.DatetimeIndex:
    """Return rebalance dates according to settings.

    Expected settings keys:
      - REBALANCE_FREQUENCY: one of {"D","W","BW","TW","M"}
      - REBALANCE_DAY: weekday (used when frequency is W/BW/TW), e.g. "FRI"

    Definitions:
      - D: every trading date
      - W: every trading date that falls on REBALANCE_DAY
      - BW: every 2nd trading date that falls on REBALANCE_DAY
      - TW: every 3rd trading date that falls on REBALANCE_DAY
      - M: end-of-month trading date
    """
    freq = str(settings.get("REBALANCE_FREQUENCY", "W")).upper()

    if freq == "D":
        return trading_dates

    if freq in {"W", "BW", "TW", "BIWEEKLY", "TRIWEEKLY"}:
        day = str(settings.get("REBALANCE_DAY", "FRI")).upper()
        if day not in WEEKDAY_MAP:
            raise ValueError(f"REBALANCE_DAY must be one of {sorted(WEEKDAY_MAP)}, got {day}")
        wd = WEEKDAY_MAP[day]
        weekday_dates = trading_dates[trading_dates.weekday == wd]

        if freq == "W":
            return weekday_dates
        if freq in {"BW", "BIWEEKLY"}:
            return weekday_dates[::2]
        if freq in {"TW", "TRIWEEKLY"}:
            return weekday_dates[::3]

    if freq == "M":
        df = pd.DataFrame({"date": trading_dates})
        df["ym"] = df["date"].dt.to_period("M")
        eom = df.groupby("ym")["date"].max().sort_values()
        return pd.DatetimeIndex(eom.values)

    raise ValueError(f"Unsupported REBALANCE_FREQUENCY: {freq} (use D/W/BW/TW/M)")


def build_cumlogret(df: pd.DataFrame, id_col: str, date_col: str, ret_col: str) -> pd.Series:
    """Cumulated log returns per asset as MultiIndex Series (id,date)->cumlog.

    Guardrail:
      log1p(ret) is undefined for ret <= -1. CRSP can contain -1 returns (e.g., delisting).
      For baseline plumbing, we clip slightly above -1 to avoid -inf.

    Returns:
      MultiIndex Series with index (id, date) and values cumlog = sum_t log1p(ret_t).
    """
    d = df[[id_col, date_col, ret_col]].copy()
    d[ret_col] = pd.to_numeric(d[ret_col], errors="coerce").astype(float)

    d["ret_clipped"] = d[ret_col].clip(lower=-0.999999999)
    d["log1p"] = np.log1p(d["ret_clipped"].fillna(0.0))

    d = d.sort_values([id_col, date_col])
    d["cumlog"] = d.groupby(id_col)["log1p"].cumsum()
    return d.set_index([id_col, date_col])["cumlog"].astype(float)


def period_asset_returns(cumlog: pd.Series, ids: pd.Index, t: pd.Timestamp, t_next: pd.Timestamp) -> pd.Series:
    """Compounded return over (t, t_next] via expm1(cumlog(t_next) - cumlog(t))."""
    idx_t = pd.MultiIndex.from_product([ids, [t]])
    idx_n = pd.MultiIndex.from_product([ids, [t_next]])

    c_t = cumlog.reindex(idx_t)
    c_n = cumlog.reindex(idx_n)

    c_t.index = c_t.index.droplevel(1)
    c_n.index = c_n.index.droplevel(1)

    lr = c_n - c_t
    out = np.expm1(lr)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_covariance_from_returns(
    rets_wide: pd.DataFrame,
    lookback_days: int,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a sample covariance matrix from trailing daily returns.

    Uses the last `lookback_days` rows up to and including `end_date`.
    Drops names with too few observations and fills remaining NaNs with 0.0.
    """
    window = rets_wide.loc[:end_date].tail(int(lookback_days))

    if window.empty:
        return pd.DataFrame(index=rets_wide.columns, columns=rets_wide.columns, dtype=float)

    min_obs = max(60, int(0.5 * lookback_days))
    counts = window.count()
    good_cols = counts[counts >= min_obs].index
    window = window.loc[:, good_cols]

    window = window.fillna(0.0)
    return window.cov()


def optimized_portfolio_weights(
    signal_today: pd.Series,
    rets_wide: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    cov_lookback_days: int,
    lam: float,
    w_max: float,
    gross_max: float,
    shrink_delta: float,
    w_prev: pd.Series | None = None,
    turnover_gamma: float = 0.0,
    beta_today: pd.Series | None = None,
    sector_today: pd.Series | None = None,
) -> pd.Series:
    """Compute optimized dollar-neutral weights using mean-variance style objective.

    Phase 8 (step 1): constraints enforced by optimizer.py:
      - sum(w)=0
      - beta^T w = 0 (only if beta_today is provided)
      - |w_i| <= w_max
      - sum(|w|) <= gross_max
      - simple shrinkage covariance

    Returns pd.Series indexed by permno.
    """
    x = signal_today.dropna()
    if x.empty:
        return pd.Series(dtype=float)

    # Convert signal to a stable optimizer alpha scale using cross-sectional ranks.
    # This avoids treating small raw signal-level differences as economically meaningful
    # alpha differences inside the optimizer objective.
    ranks = x.rank(method="average")
    mu = (ranks - ranks.mean()) / (ranks.std(ddof=0) + 1e-12)

    Sigma = build_covariance_from_returns(
        rets_wide=rets_wide,
        lookback_days=int(cov_lookback_days),
        end_date=pd.Timestamp(rebalance_date),
    )

    if Sigma.empty or Sigma.shape[0] < 2:
        return pd.Series(dtype=float)

    params = OptimizerParams(
        lam=float(lam),
        turnover_gamma=float(turnover_gamma),
        w_max=float(w_max),
        gross_max=float(gross_max),
        shrink_delta=float(shrink_delta),
    )

    w = optimize_mean_variance(
        mu=mu,
        Sigma=Sigma,
        beta=beta_today,
        sector=sector_today,
        w_prev=w_prev,
        params=params,
    )
    return w


@dataclass
class BacktestResult:
    period_end_dates: pd.DatetimeIndex
    period_returns: pd.Series
    period_details: pd.DataFrame
    turnover: pd.Series
    equity_curve: pd.Series
    metrics: Dict[str, float]
    weights: pd.DataFrame


def run_baseline_backtest(
    df: pd.DataFrame,
    settings: Dict[str, Any],
    signal_col: str,
    universe_mask: pd.DataFrame | None = None,
    id_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret_total",
) -> BacktestResult:
    """Run baseline long/short quantile backtest.

    Mechanics:
      - On each rebalance date t: rank signal cross-sectionally and form equal-weight long/short
      - Hold weights constant until next rebalance date t_next
      - Compute holding-period return per asset over (t, t_next] using compounded daily returns
      - Portfolio return for the period is sum_i w_i,t * R_i,(t,t_next]
      - Turnover at period end is sum_i |w_t - w_prev|

    Output series are indexed by period end date (t_next).
    """
    # Base columns required for all runs
    cols = [id_col, date_col, ret_col, signal_col]

    # Phase 9: optional transaction costs / execution constraints (all OFF by default)
    apply_spread_cost = bool(settings.get("APPLY_SPREAD_COST", False))
    apply_vol_slippage = bool(settings.get("APPLY_VOL_SLIPPAGE", False))
    apply_participation = bool(settings.get("APPLY_PARTICIPATION_CONSTRAINT", False))
    apply_turnover_cost = bool(settings.get("APPLY_TURNOVER_COST", False))

    # Columns (case-insensitive fallbacks)
    bid_col = resolve_col(df, "bid", "BID")
    ask_col = resolve_col(df, "ask", "ASK")
    bidlo_col = resolve_col(df, "bidlo", "BIDLO")
    askhi_col = resolve_col(df, "askhi", "ASKHI")

    prc_col = resolve_col(df, "prc", "PRC")
    vol_col = resolve_col(df, "vol", "VOL")

    if apply_spread_cost:
        missing = [
            name
            for name, col in [("bid", bid_col), ("ask", ask_col), ("bidlo", bidlo_col), ("askhi", askhi_col)]
            if col is None
        ]
        if missing:
            raise KeyError(
                "APPLY_SPREAD_COST=True but missing required quote columns: "
                + ", ".join(missing)
                + ". Expected bid/ask and bidlo/askhi (case-insensitive)."
            )
        cols.extend([bid_col, ask_col, bidlo_col, askhi_col])

    if apply_participation:
        missing = [name for name, col in [("prc", prc_col), ("vol", vol_col)] if col is None]
        if missing:
            raise KeyError(
                "APPLY_PARTICIPATION_CONSTRAINT=True but missing required columns: "
                + ", ".join(missing)
                + ". Expected prc and vol (case-insensitive)."
            )
        cols.extend([prc_col, vol_col])

    # Optional optimizer exposures (generic wiring). We only add columns that exist.
    beta_col = str(settings.get("OPT_BETA_COL", "beta"))
    sector_col = str(settings.get("OPT_SECTOR_COL", "sector"))

    if beta_col in df.columns:
        cols.append(beta_col)
    if sector_col in df.columns:
        cols.append(sector_col)

    d = df[cols].copy()

    d[date_col] = pd.to_datetime(d[date_col])

    spread_lookup = None
    rv_lookup = None
    dadv_lookup = None

    if apply_spread_cost:
        spread_df = compute_spread_proxy(
            d,
            id_col=id_col,
            date_col=date_col,
            bid_col=bid_col,
            ask_col=ask_col,
            bidlo_col=bidlo_col,
            askhi_col=askhi_col,
            cap=float(settings.get("SPREAD_CAP", 0.20)),
        )
        spread_lookup = spread_df.set_index([date_col, id_col])["qspread_lag1"].astype(float)

    if apply_vol_slippage:
        rv_df = compute_realized_vol_proxy(
            d,
            id_col=id_col,
            date_col=date_col,
            ret_col=ret_col,
            lookback=int(settings.get("VOL_SLIP_LOOKBACK", 20)),
            cap=float(settings.get("VOL_CAP", 0.20)),
        )
        rv_lookup = rv_df.set_index([date_col, id_col])["rv_lag1"].astype(float)

    if apply_participation:
        adv_df = compute_dollar_adv_proxy(
            d,
            id_col=id_col,
            date_col=date_col,
            prc_col=prc_col,
            vol_col=vol_col,
            lookback=int(settings.get("ADV_LOOKBACK", 20)),
        )
        dadv_lookup = adv_df.set_index([date_col, id_col])["dadv_lag1"].astype(float)

    # Phase 8: optional optimizer path (keeps baseline as default)
    use_optimizer = bool(settings.get("USE_OPTIMIZER", False))
    opt_cov_lookback = int(settings.get("OPT_COV_LOOKBACK_DAYS", 252))
    opt_shrink_delta = float(settings.get("OPT_SHRINK_DELTA", 0.2))
    opt_lam = float(settings.get("OPT_LAM", 1.0))
    opt_w_max = float(settings.get("OPT_W_MAX", 0.01))
    opt_gross_max = float(settings.get("OPT_GROSS_MAX", 2.0))
    opt_turnover_gamma = float(settings.get("OPT_TURNOVER_GAMMA", 0.0))

    opt_beta_neutral = bool(settings.get("OPT_BETA_NEUTRAL", False))
    # Sector neutrality flag is wired generically; may be enabled later.
    opt_sector_neutral = bool(settings.get("OPT_SECTOR_NEUTRAL", False))

    if use_optimizer and opt_beta_neutral and beta_col not in d.columns:
        raise KeyError(f"OPT_BETA_NEUTRAL=True but beta column '{beta_col}' not found in df.")
    if use_optimizer and opt_sector_neutral and sector_col not in d.columns:
        raise KeyError(f"OPT_SECTOR_NEUTRAL=True but sector column '{sector_col}' not found in df.")

    # Wide daily returns matrix for covariance estimation (index=date, cols=permno)
    rets_wide = d.pivot(index=date_col, columns=id_col, values=ret_col).sort_index()

    trading_dates = pd.DatetimeIndex(sorted(d[date_col].unique()))
    reb_dates = get_rebalance_dates(trading_dates, settings)
    if len(reb_dates) < 2:
        raise ValueError("Not enough rebalance dates in sample to run backtest.")

    cumlog = build_cumlogret(d, id_col=id_col, date_col=date_col, ret_col=ret_col)

    sig = d[[id_col, date_col, signal_col]].dropna(subset=[signal_col]).copy()
    sig = sig.set_index([date_col, id_col])[signal_col].astype(float)

    # Optional exposure lookups (date, id) -> exposure value
    beta_lookup = None
    if use_optimizer and opt_beta_neutral:
        if beta_col not in d.columns:
            raise KeyError(f"OPT_BETA_NEUTRAL=True but beta column '{beta_col}' not found in df.")
        b = d[[id_col, date_col, beta_col]].copy()
        b[beta_col] = pd.to_numeric(b[beta_col], errors="coerce")
        beta_lookup = b.set_index([date_col, id_col])[beta_col].astype(float)

    sector_lookup = None
    if sector_col in d.columns:
        s = d[[id_col, date_col, sector_col]].dropna(subset=[sector_col]).copy()
        sector_lookup = s.set_index([date_col, id_col])[sector_col]

    # Optional universe mask: DataFrame with columns [date, permno, in_universe].
    # Applied at each rebalance date *before* cross-sectional ranking.
    mask = None
    if universe_mask is not None:
        mask = universe_mask[[date_col, id_col, "in_universe"]].copy()
        mask[date_col] = pd.to_datetime(mask[date_col])
        mask["in_universe"] = mask["in_universe"].astype("boolean").fillna(False).astype(bool)
        # Ensure uniqueness to avoid accidental duplication on merge/filter.
        mask = mask.drop_duplicates(subset=[date_col, id_col])
        mask = mask.set_index([date_col, id_col])["in_universe"]

    long_q = float(settings.get("LONG_QUANTILE", 0.9))
    short_q = float(settings.get("SHORT_QUANTILE", 0.1))

    date_to_pos = {dt: i for i, dt in enumerate(trading_dates)}

    period_end_dates: List[pd.Timestamp] = []
    period_returns: List[float] = []
    turnovers: List[float] = []
    period_details_rows: List[Dict[str, float | pd.Timestamp | bool]] = []
    weights_frames: List[pd.DataFrame] = []

    w_prev = pd.Series(dtype=float)
    w_prev_use = pd.Series(dtype=float)

    show_progress = bool(settings.get("SHOW_PROGRESS", False))
    _iter = zip(reb_dates[:-1], reb_dates[1:])
    if show_progress and (tqdm is not None):
        _iter = tqdm(_iter, total=max(len(reb_dates) - 1, 0), desc="engine", unit="reb")

    for t, t_next in _iter:
        try:
            s_t = sig.loc[t]
        except KeyError:
            continue

        # Apply universe mask at formation date (rebalance date).
        if mask is not None:
            try:
                in_u = mask.loc[t]
            except KeyError:
                continue
            eligible = in_u[in_u].index
            if len(eligible) == 0:
                continue
            s_t = s_t.reindex(eligible).dropna()
            if s_t.empty:
                continue

        if not use_optimizer:
            long_ids, short_ids = rank_to_long_short(s_t, long_q=long_q, short_q=short_q)
            w_t = make_equal_ls_weights(long_ids, short_ids)
            if w_t.empty:
                continue
        else:
            beta_t = None
            if opt_beta_neutral and beta_lookup is not None:
                try:
                    beta_t = beta_lookup.loc[t].reindex(s_t.index)
                except KeyError:
                    beta_t = None

                # If beta neutrality is requested but beta is missing, skip this rebalance.
                if beta_t is None:
                    continue

                # Filter optimizer universe to names with non-missing beta.
                good = beta_t.notna()
                n_nonnull = int(good.sum())
                n_total = int(len(beta_t))
                min_required = max(20, int(0.5 * n_total))
                if n_nonnull < min_required:
                    continue

                s_t = s_t.loc[good[good].index]
                beta_t = beta_t.loc[s_t.index].astype(float)

            sector_t = None
            if opt_sector_neutral and sector_lookup is not None:
                try:
                    sector_t = sector_lookup.loc[t].reindex(s_t.index)
                except KeyError:
                    sector_t = None

                # If sector neutrality is requested but sector is missing, skip this rebalance.
                if sector_t is None:
                    continue

                # Filter optimizer universe to names with non-missing sector.
                good_s = sector_t.notna()
                n_nonnull_s = int(good_s.sum())
                n_total_s = int(len(sector_t))
                min_required_s = max(20, int(0.5 * n_total_s))
                if n_nonnull_s < min_required_s:
                    continue

                s_t = s_t.loc[good_s[good_s].index]
                if beta_t is not None:
                    beta_t = beta_t.loc[s_t.index]
                sector_t = sector_t.loc[s_t.index]

            w_t = optimized_portfolio_weights(
                signal_today=s_t,
                rets_wide=rets_wide,
                rebalance_date=t,
                cov_lookback_days=opt_cov_lookback,
                lam=opt_lam,
                w_max=opt_w_max,
                gross_max=opt_gross_max,
                shrink_delta=opt_shrink_delta,
                w_prev=w_prev if not w_prev.empty else None,
                turnover_gamma=opt_turnover_gamma,
                beta_today=beta_t,
                sector_today=sector_t,
            )
            if w_t is None or w_t.empty:
                continue

        to = simple_turnover(w_prev, w_t)

        ids = w_t.index
        r_i = period_asset_returns(cumlog=cumlog, ids=ids, t=t, t_next=t_next)

        if not use_optimizer:
            # Baseline path: drop missing returns and renormalize
            valid = r_i.dropna()
            if valid.empty:
                continue

            w_use = w_t.reindex(valid.index)

            longs = w_use[w_use > 0]
            shorts = w_use[w_use < 0]
            if len(longs) == 0 or len(shorts) == 0:
                continue

            # Equal-weight within each side
            w_use.loc[longs.index] = 1.0 / len(longs)
            w_use.loc[shorts.index] = -1.0 / len(shorts)

            rp = float((w_use * valid).sum())

        else:
            # Optimizer path: preserve optimizer weights and constraints
            # Fill missing asset returns with 0 to avoid renormalization
            r_i = r_i.fillna(0.0)
            w_use = w_t.copy()
            rp = float((w_use * r_i).sum())

        # --- Phase 9: Execution constraint (participation) and transaction costs ---
        # All models use lagged (t-1) inputs at execution date t to avoid lookahead.

        # Returns vector used to compute rp (depends on optimizer path)
        if not use_optimizer:
            ret_vec = valid
        else:
            ret_vec = r_i

        idx = w_use.index.union(w_prev_use.index)

        # Desired trade from previous used weights to target used weights
        w_prev_aligned = w_prev_use.reindex(idx).fillna(0.0)
        w_tgt_aligned = w_use.reindex(idx).fillna(0.0)

        # Apply participation constraint by clipping trade weights (optional)
        if apply_participation and dadv_lookup is not None:
            try:
                dadv = dadv_lookup.loc[t].reindex(idx)
            except KeyError:
                dadv = pd.Series(index=idx, dtype=float)

            w_exec, dw_exec = apply_participation_constraint_to_weights(
                w_prev=w_prev_aligned,
                w_target=w_tgt_aligned,
                dadv_lag1=dadv,
                portfolio_value=float(settings.get("PORTFOLIO_VALUE", 1_000_000.0)),
                max_participation=float(settings.get("MAX_PARTICIPATION", 0.10)),
            )
        else:
            dw_exec = w_tgt_aligned - w_prev_aligned
            w_exec = w_tgt_aligned

        # Recompute rp using executed weights (uninvested notional is allowed when constrained)
        rp = float((w_exec.reindex(ret_vec.index).fillna(0.0) * ret_vec).sum())

        # Apply costs on executed trades
        gross_rp = float(rp)
        spread_cost_value = 0.0
        vol_slippage_cost_value = 0.0
        turnover_cost_value = 0.0
        total_cost = 0.0

        if apply_spread_cost and spread_lookup is not None:
            try:
                qlag = spread_lookup.loc[t].reindex(idx)
            except KeyError:
                qlag = pd.Series(index=idx, dtype=float)
            qlag = pd.to_numeric(qlag, errors="coerce").fillna(0.0).clip(lower=0.0, upper=float(settings.get("SPREAD_CAP", 0.20)))
            spread_cost_value = float(spread_cost_from_dw(dw_exec, qlag))
            total_cost += spread_cost_value

        if apply_vol_slippage and rv_lookup is not None:
            try:
                rvlag = rv_lookup.loc[t].reindex(idx)
            except KeyError:
                rvlag = pd.Series(index=idx, dtype=float)
            rvlag = pd.to_numeric(rvlag, errors="coerce").fillna(0.0).clip(lower=0.0, upper=float(settings.get("VOL_CAP", 0.20)))
            vol_slippage_cost_value = float(
                vol_slippage_cost_from_dw(dw_exec, rvlag, k=float(settings.get("VOL_SLIP_K", 0.10)))
            )
            total_cost += vol_slippage_cost_value

        if apply_turnover_cost:
            turnover_cost_value = float(
                turnover_cost_from_dw(
                    dw_exec,
                    cost_per_dollar=float(settings.get("TURNOVER_COST_PER_DOLLAR", 0.0)),
                )
            )
            total_cost += turnover_cost_value

        rp = float(gross_rp - total_cost)

        # Keep executed weights as the weights actually used for this holding period
        w_use = w_exec

        # Save weights actually used for this holding period
        w_snap = w_use.dropna().copy()
        if not w_snap.empty:
            weights_frames.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(t),
                        id_col: w_snap.index.astype(int),
                        "weight": w_snap.values.astype(float),
                    }
                )
            )

        period_end_dates.append(pd.Timestamp(t_next))
        period_returns.append(rp)
        turnovers.append(to)
        period_details_rows.append(
            {
                "formation_date": pd.Timestamp(t),
                "date": pd.Timestamp(t_next),
                "gross_return": float(gross_rp),
                "net_return": float(rp),
                "turnover": float(to),
                "spread_cost": float(spread_cost_value),
                "vol_slippage_cost": float(vol_slippage_cost_value),
                "turnover_cost": float(turnover_cost_value),
                "total_cost": float(total_cost),
                "gross_exposure_used": float(w_use.abs().sum()),
                "net_exposure_used": float(w_use.sum()),
                "n_holdings": int(w_use.replace(0.0, np.nan).dropna().shape[0]),
                "participation_constrained": bool(apply_participation and not w_exec.equals(w_tgt_aligned)),
            }
        )

        w_prev = w_t
        w_prev_use = w_use.copy()

    if len(period_returns) == 0:
        raise ValueError("Backtest produced zero periods. Check signal availability and rebalance settings.")

    pr = pd.Series(period_returns, index=pd.DatetimeIndex(period_end_dates), name="portfolio_return")
    to_ser = pd.Series(turnovers, index=pd.DatetimeIndex(period_end_dates), name="turnover")

    eq = compute_equity_curve(pr)

    # Annualization using average trading-day length between period ends
    pos = [date_to_pos.get(dt) for dt in pr.index]
    pos = [p for p in pos if p is not None]
    if len(pos) >= 2:
        deltas = np.diff(pos)
        avg_days = float(np.mean(deltas)) if len(deltas) > 0 else 5.0
    else:
        avg_days = 5.0
    periods_per_year = 252.0 / max(avg_days, 1.0)

    metrics = summarize_performance(period_returns=pr, periods_per_year=periods_per_year, turnover=to_ser)

    if weights_frames:
        weights_df = pd.concat(weights_frames, ignore_index=True)
        weights_df["date"] = pd.to_datetime(weights_df["date"])
    else:
        weights_df = pd.DataFrame(columns=["date", id_col, "weight"])

    if period_details_rows:
        period_details_df = pd.DataFrame(period_details_rows)
        period_details_df["formation_date"] = pd.to_datetime(period_details_df["formation_date"])
        period_details_df["date"] = pd.to_datetime(period_details_df["date"])
    else:
        period_details_df = pd.DataFrame(
            columns=[
                "formation_date",
                "date",
                "gross_return",
                "net_return",
                "turnover",
                "spread_cost",
                "vol_slippage_cost",
                "turnover_cost",
                "total_cost",
                "gross_exposure_used",
                "net_exposure_used",
                "n_holdings",
                "participation_constrained",
            ]
        )

    return BacktestResult(
        period_end_dates=pr.index,
        period_returns=pr,
        period_details=period_details_df,
        turnover=to_ser,
        equity_curve=eq,
        metrics=metrics,
        weights=weights_df,
    )

