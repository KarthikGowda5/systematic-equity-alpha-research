"""Factor exposure regression (Phase 10).

Goal
----
Estimate systematic factor exposures of the realized portfolio return series via
linear regression on common equity factors:
    Market, SMB, HML, Momentum

This module is *analysis-only*: it does not modify portfolio construction.

Data contract
------------
- Portfolio returns: pd.Series indexed by date with periodic returns (daily)
- Factor returns: loaded from a local CSV (recommended for reproducibility)

Supported factor schemas
------------------------
The loader attempts to standardize common column conventions.
It accepts any of these (case-insensitive) column sets:
- Market:   'MKT', 'MKT_RF', 'MKT-RF', 'Mkt-RF', 'MARKET', 'RM'
- SMB:      'SMB'
- HML:      'HML'
- Momentum: 'MOM', 'MOMENTUM', 'UMD', 'Mom'
- Risk-free (optional): 'RF', 'R_F', 'RISKFREE'

Notes
-----
- If the factor file contains 'MKT-RF' and 'RF', we regress *excess* portfolio
  returns (r - rf) on the excess market factor.
- If there is no RF column, we regress raw portfolio returns on factors.
- Standard errors default to Newey-West (HAC) with lag=5 to match your IC
  convention, but can be disabled (nw_lags=0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRegressionConfig:
    # Path to a local factor CSV. You can keep this in config.py and pass it in.
    factors_csv: str = "data/raw/ff_factors_mom_daily.csv"

    date_col: str = "date"

    # Newey-West HAC lags; set to 0 for plain OLS SEs
    nw_lags: int = 5

    # Annualization is NOT needed for betas, but may be useful for alpha display.
    ann_factor: int = 252


# ----------------------------
# Loading / standardization
# ----------------------------

def _canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def load_factor_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    """Load factors from CSV and standardize column names.

    Returns a DataFrame indexed by datetime with columns:
        mkt, smb, hml, mom, rf (rf may be absent)
    """
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Factor CSV missing date column '{date_col}'. Found: {list(df.columns)[:20]}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Build mapping from canonical names
    cols = {c: _canon(c) for c in df.columns}

    def pick(*cands: str) -> Optional[str]:
        for cand in cands:
            cc = _canon(cand)
            for orig, can in cols.items():
                if can == cc:
                    return orig
        return None

    mkt = pick("mkt", "market", "rm", "mktrf", "mktrf", "mktrf")
    mkt_rf = pick("mktrf", "mktrf", "mktrf", "mktrf", "mktrf")
    # Explicit match for 'MKT-RF' patterns
    for orig, can in cols.items():
        if can in {"mktrf", "mktrf", "mktrf"}:
            mkt_rf = orig

    smb = pick("smb")
    hml = pick("hml")
    mom = pick("mom", "momentum", "umd")
    rf = pick("rf", "riskfree", "r_f")

    # Prefer MKT-RF if present
    mkt_col = mkt_rf if mkt_rf is not None else mkt

    required = {"mkt": mkt_col, "smb": smb, "hml": hml, "mom": mom}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Factor CSV missing required columns for: "
            + ", ".join(missing)
            + ". Found columns: "
            + ", ".join(df.columns)
        )

    out = pd.DataFrame(index=df.index)
    out["mkt"] = pd.to_numeric(df[required["mkt"]], errors="coerce")
    out["smb"] = pd.to_numeric(df[required["smb"]], errors="coerce")
    out["hml"] = pd.to_numeric(df[required["hml"]], errors="coerce")
    out["mom"] = pd.to_numeric(df[required["mom"]], errors="coerce")
    if rf is not None:
        out["rf"] = pd.to_numeric(df[rf], errors="coerce")

    out = out.dropna(how="all")

    # Heuristic: if factors look like percent units (e.g., 0.12 meaning 0.12%),
    # we leave them as-is; user should supply consistent units with portfolio.
    return out


# ----------------------------
# Regression core
# ----------------------------

def _to_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    r = returns.copy()
    if not isinstance(r.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            r.index = pd.to_datetime(r.index)
        except Exception as e:
            raise TypeError("returns index must be datetime-like or convertible") from e

    r = pd.to_numeric(r, errors="coerce").astype(float).dropna().sort_index()
    return r


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Stable least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _nw_cov(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance for OLS coefficients.

    X: (T, k) with intercept included
    resid: (T,) residuals
    """
    T, k = X.shape
    # Meat: sum_{t} x_t x_t' e_t^2 + weighted lag terms
    S = np.zeros((k, k), dtype=float)

    # t=0..T-1
    for t in range(T):
        xt = X[t : t + 1].T  # (k,1)
        S += (resid[t] ** 2) * (xt @ xt.T)

    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Sl = np.zeros((k, k), dtype=float)
        for t in range(L, T):
            xt = X[t : t + 1].T
            xtL = X[t - L : t - L + 1].T
            Sl += resid[t] * resid[t - L] * (xt @ xtL.T)
        S += w * (Sl + Sl.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    return cov


def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    nw_lags: int = 5,
) -> Dict[str, float]:
    """Regress portfolio returns on factors.

    returns: pd.Series
    factors: DataFrame indexed by date with columns mkt,smb,hml,mom,(optional rf)

    Returns a flat dict with alpha/betas, t-stats, r2, and n_obs.
    """
    r = _to_series(returns)

    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame")

    f = factors.copy()
    if not isinstance(f.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            f.index = pd.to_datetime(f.index)
        except Exception as e:
            raise TypeError("factors index must be datetime-like or convertible") from e

    f = f.sort_index()

    # Align
    df = pd.concat([r.rename("ret"), f], axis=1, join="inner").dropna()
    if df.shape[0] < 20:
        # Too few obs to be meaningful
        return {
            "n_obs": float(df.shape[0]),
            "alpha": float("nan"),
            "beta_mkt": float("nan"),
            "beta_smb": float("nan"),
            "beta_hml": float("nan"),
            "beta_mom": float("nan"),
            "t_alpha": float("nan"),
            "t_mkt": float("nan"),
            "t_smb": float("nan"),
            "t_hml": float("nan"),
            "t_mom": float("nan"),
            "r2": float("nan"),
        }

    y = df["ret"].values.astype(float)

    # Excess return handling
    if "rf" in df.columns and "mkt" in df.columns:
        # If mkt is an excess market factor, typical factor file will use MKT-RF as mkt.
        # We regress excess portfolio return.
        y = y - df["rf"].values.astype(float)

    Xf = df[["mkt", "smb", "hml", "mom"]].values.astype(float)
    T = Xf.shape[0]

    # Add intercept
    X = np.column_stack([np.ones(T), Xf])

    beta = _ols_beta(X, y)
    y_hat = X @ beta
    resid = y - y_hat

    # R^2
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    # Covariance / SEs
    if nw_lags and nw_lags > 0:
        cov = _nw_cov(X, resid, int(nw_lags))
    else:
        # Homoskedastic OLS covariance
        XtX_inv = np.linalg.inv(X.T @ X)
        sigma2 = sse / (T - X.shape[1])
        cov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(cov))
    se = np.where(se == 0, np.nan, se)

    tstats = beta / se

    out: Dict[str, float] = {
        "n_obs": float(T),
        "alpha": float(beta[0]),
        "beta_mkt": float(beta[1]),
        "beta_smb": float(beta[2]),
        "beta_hml": float(beta[3]),
        "beta_mom": float(beta[4]),
        "t_alpha": float(tstats[0]),
        "t_mkt": float(tstats[1]),
        "t_smb": float(tstats[2]),
        "t_hml": float(tstats[3]),
        "t_mom": float(tstats[4]),
        "r2": float(r2),
    }

    return out


def compute_factor_exposures(
    returns: pd.Series,
    config: Optional[FactorRegressionConfig] = None,
) -> Dict[str, float]:
    """Convenience wrapper: load factors from CSV and run regression."""
    cfg = config or FactorRegressionConfig()
    factors = load_factor_csv(cfg.factors_csv, cfg.date_col)
    return run_factor_regression(returns, factors, nw_lags=cfg.nw_lags)
