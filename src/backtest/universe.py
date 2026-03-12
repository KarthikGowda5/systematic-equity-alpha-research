

"""Universe construction utilities.

Phase 5 adds a liquidity filter (monthly reconstitution by default).

This module builds a *date-by-date eligibility mask* that can be applied
before cross-sectional ranking / portfolio construction.

Design goals:
- Deterministic, reproducible universe membership
- Works with daily CRSP-style panels: (permno, date, dollar_vol, mktcap)
- Reconstitution dates are chosen as the *last available trading date* in each
  period (month/week) based on the dates present in the input data.

Notes
-----
- This module does *not* apply exchange/share-code filters because those
  columns may not exist in your processed panel. If you have them later, you
  can apply those filters upstream before calling these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


LiquidityFilterType = Literal["ADV", "MARKET_CAP"]
ReconFreq = Literal["D", "W", "M"]


@dataclass(frozen=True)
class UniverseSpec:
    """Specification for building a liquidity-based trading universe."""

    filter_type: LiquidityFilterType = "ADV"
    top_n: int = 1000
    recon_freq: ReconFreq = "M"  # D=every day, W=weekly, M=monthly
    adv_window: int = 20  # trading days for ADV (rolling mean of dollar volume)


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for universe construction: {missing}")


def get_reconstitution_dates(dates: pd.Series | pd.DatetimeIndex, recon_freq: ReconFreq) -> pd.DatetimeIndex:
    """Return reconstitution dates as the last available date in each period.

    Parameters
    ----------
    dates:
        All available trading dates in the panel.
    recon_freq:
        'D' daily (every date), 'W' weekly (last date in each ISO week),
        'M' monthly (last date in each month).

    Returns
    -------
    pd.DatetimeIndex
        Sorted unique reconstitution dates.
    """
    if isinstance(dates, pd.Series):
        d = pd.to_datetime(dates)
    else:
        d = pd.to_datetime(pd.Series(dates))

    d = d.dropna().drop_duplicates().sort_values()
    if d.empty:
        return pd.DatetimeIndex([])

    if recon_freq == "D":
        return pd.DatetimeIndex(d.values)

    # Group by period and take max date available in that period.
    if recon_freq == "M":
        key = d.dt.to_period("M")
    elif recon_freq == "W":
        # ISO week period; using W-SUN gives stable weekly buckets.
        key = d.dt.to_period("W-SUN")
    else:
        raise ValueError(f"Unsupported recon_freq: {recon_freq}")

    recon = d.groupby(key).max().sort_values()
    return pd.DatetimeIndex(recon.values)


def compute_liquidity_score(panel: pd.DataFrame, spec: UniverseSpec) -> pd.DataFrame:
    """Compute per-(permno,date) liquidity score used for ranking.

    For ADV: rolling mean of dollar volume over `adv_window` trading days.
    For MARKET_CAP: uses `mktcap` as-of that date.

    Returns
    -------
    pd.DataFrame
        Columns: permno, date, score
    """
    _require_cols(panel, ["permno", "date"])
    df = panel[["permno", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    if spec.filter_type == "ADV":
        _require_cols(panel, ["dollar_vol"])
        df["dollar_vol"] = panel["dollar_vol"].astype(float)
        df = df.sort_values(["permno", "date"])
        # Rolling mean within permno; min_periods ensures early history isn't all NaN.
        df["score"] = (
            df.groupby("permno", sort=False)["dollar_vol"]
            .rolling(window=spec.adv_window, min_periods=max(5, spec.adv_window // 4))
            .mean()
            .reset_index(level=0, drop=True)
        )
    elif spec.filter_type == "MARKET_CAP":
        _require_cols(panel, ["mktcap"])
        df["score"] = panel["mktcap"].astype(float)
    else:
        raise ValueError(f"Unsupported filter_type: {spec.filter_type}")

    return df[["permno", "date", "score"]]


def build_liquidity_universe_mask(panel: pd.DataFrame, spec: UniverseSpec) -> pd.DataFrame:
    """Build a date-by-date universe membership mask.

    Membership is determined on reconstitution dates by ranking liquidity score
    and selecting the top `spec.top_n` names.

    For any date t, the active universe is defined by the most recent
    reconstitution date r <= t.

    Parameters
    ----------
    panel:
        Daily panel with at least: permno, date and either dollar_vol (ADV) or mktcap.
    spec:
        UniverseSpec.

    Returns
    -------
    pd.DataFrame
        Columns: date, permno, in_universe (bool)
    """
    _require_cols(panel, ["permno", "date"])
    if spec.top_n <= 0:
        raise ValueError("top_n must be positive")

    df = panel[["permno", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["permno", "date"]).copy()

    # Compute scores across all days.
    score_df = compute_liquidity_score(panel, spec)

    all_dates = pd.DatetimeIndex(df["date"].drop_duplicates().sort_values().values)
    recon_dates = get_reconstitution_dates(all_dates, spec.recon_freq)
    if len(recon_dates) == 0:
        out = df.copy()
        out["in_universe"] = False
        return out[["date", "permno", "in_universe"]]

    # Create a mapping from each date to its most recent recon date.
    date_map = pd.DataFrame({"date": all_dates})
    recon_map = pd.DataFrame({"recon_date": recon_dates})
    # merge_asof requires sorted
    date_map = date_map.sort_values("date")
    recon_map = recon_map.sort_values("recon_date")
    date_map = pd.merge_asof(
        date_map,
        recon_map,
        left_on="date",
        right_on="recon_date",
        direction="backward",
        allow_exact_matches=True,
    )
    # If earliest dates precede first recon date (shouldn't happen), drop them.
    date_map = date_map.dropna(subset=["recon_date"]).copy()

    # Build membership at each recon date.
    s_at_recon = score_df[score_df["date"].isin(recon_dates)].copy()
    # Drop NaN scores (e.g., insufficient ADV history)
    s_at_recon = s_at_recon.dropna(subset=["score"]).copy()

    # Rank within each recon date and pick top N.
    s_at_recon["rank"] = s_at_recon.groupby("date")["score"].rank(method="first", ascending=False)
    top = s_at_recon[s_at_recon["rank"] <= spec.top_n][["date", "permno"]].copy()
    top = top.rename(columns={"date": "recon_date"})
    top["in_universe"] = True

    # Expand membership to all dates via (date -> recon_date) mapping.
    expanded = df[["date", "permno"]].drop_duplicates().copy()
    expanded = expanded.merge(date_map, on="date", how="left")
    expanded = expanded.merge(top, on=["recon_date", "permno"], how="left")
    # Ensure stable boolean dtype without pandas' silent downcasting warnings.
    expanded["in_universe"] = expanded["in_universe"].astype("boolean").fillna(False).astype(bool)

    return expanded[["date", "permno", "in_universe"]]


def apply_universe_mask(panel: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper to filter a panel using a (date,permno)->in_universe mask."""
    _require_cols(panel, ["date", "permno"])
    _require_cols(mask, ["date", "permno", "in_universe"])

    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    m = mask.copy()
    m["date"] = pd.to_datetime(m["date"])

    out = p.merge(m[["date", "permno", "in_universe"]], on=["date", "permno"], how="left")
    # Ensure stable boolean dtype without pandas' silent downcasting warnings.
    out["in_universe"] = out["in_universe"].astype("boolean").fillna(False).astype(bool)
    return out[out["in_universe"]].drop(columns=["in_universe"])