

"""Value at Risk (VaR) and Expected Shortfall (ES) (Phase 10).

Implements:
- Historical VaR/ES
- Parametric (Normal) VaR/ES
- VaR backtest via breach counting

Inputs:
- returns: pd.Series of periodic portfolio returns (typically daily), indexed by date

Notes:
- VaR/ES are returned as *return thresholds* (negative numbers in typical cases).
- Breach is defined as return < VaR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VarConfig:
    levels: Tuple[float, ...] = (0.95, 0.99)
    # For parametric VaR/ES, allow user to override distribution assumptions later;
    # for Phase 10 we assume normal.


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


def historical_var_es(returns: pd.Series, level: float = 0.95) -> Dict[str, float]:
    """Compute historical VaR/ES at given confidence level.

    VaR level is the left-tail quantile (e.g., 5% for 95% VaR).
    ES is the average return conditional on being in the tail beyond VaR.
    """
    r = _to_series(returns)
    if r.empty:
        return {"var": float("nan"), "es": float("nan")}

    alpha = 1.0 - float(level)
    var = float(np.quantile(r.values, alpha, method="linear"))

    tail = r[r < var]
    es = float(tail.mean()) if tail.size > 0 else float("nan")

    return {"var": var, "es": es}


def parametric_var_es(returns: pd.Series, level: float = 0.95) -> Dict[str, float]:
    """Compute parametric (Normal) VaR/ES at given confidence level."""
    r = _to_series(returns)
    if r.size < 2:
        return {"var": float("nan"), "es": float("nan"), "mu": float("nan"), "sigma": float("nan")}

    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma == 0 or np.isnan(sigma):
        return {"var": float("nan"), "es": float("nan"), "mu": mu, "sigma": sigma}

    # z for left tail alpha
    alpha = 1.0 - float(level)

    # Use scipy if available; otherwise approximate using numpy erfinv.
    # We avoid requiring scipy as a dependency.
    z = float(np.sqrt(2.0) * _erfinv(2.0 * alpha - 1.0))  # quantile of standard normal

    var = mu + z * sigma

    # ES for normal (left tail): mu - sigma * phi(z)/alpha
    phi = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
    es = mu - sigma * (phi / alpha)

    return {"var": float(var), "es": float(es), "mu": mu, "sigma": sigma, "z": z}


def var_breaches(returns: pd.Series, var_threshold: float) -> pd.Series:
    """Boolean series indicating VaR breaches (return < VaR threshold)."""
    r = _to_series(returns)
    breaches = r < float(var_threshold)
    breaches.name = "var_breach"
    return breaches


def backtest_var_breaches(returns: pd.Series, var_threshold: float) -> Dict[str, float]:
    """Backtest VaR by counting breaches and breach rate."""
    r = _to_series(returns)
    if r.empty:
        return {"n_obs": 0.0, "n_breaches": 0.0, "breach_rate": float("nan")}

    b = var_breaches(r, var_threshold)
    n = float(r.shape[0])
    nb = float(b.sum())
    return {"n_obs": n, "n_breaches": nb, "breach_rate": (nb / n) if n > 0 else float("nan")}


def compute_var_es(
    returns: pd.Series,
    config: Optional[VarConfig] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute VaR/ES for configured confidence levels.

    Returns a nested dict:
        out["hist"]["var_95"], out["hist"]["es_95"], ...
        out["param"]["var_95"], out["param"]["es_95"], ...
        out["backtest"]["breach_rate_95_hist"], etc.
    """
    cfg = config or VarConfig()
    r = _to_series(returns)

    out: Dict[str, Dict[str, float]] = {"hist": {}, "param": {}, "backtest": {}}

    for lvl in cfg.levels:
        pct = int(round(lvl * 100))

        h = historical_var_es(r, lvl)
        out["hist"][f"var_{pct}"] = h["var"]
        out["hist"][f"es_{pct}"] = h["es"]

        p = parametric_var_es(r, lvl)
        out["param"][f"var_{pct}"] = p["var"]
        out["param"][f"es_{pct}"] = p["es"]
        out["param"][f"mu_{pct}"] = p.get("mu", float("nan"))
        out["param"][f"sigma_{pct}"] = p.get("sigma", float("nan"))
        out["param"][f"z_{pct}"] = p.get("z", float("nan"))

        # Backtest breaches for historical VaR and parametric VaR
        bt_h = backtest_var_breaches(r, h["var"])
        out["backtest"][f"breach_rate_{pct}_hist"] = bt_h["breach_rate"]
        out["backtest"][f"n_breaches_{pct}_hist"] = bt_h["n_breaches"]
        out["backtest"][f"n_obs_{pct}_hist"] = bt_h["n_obs"]

        bt_p = backtest_var_breaches(r, p["var"]) if not np.isnan(p["var"]) else {"n_obs": float(r.shape[0]), "n_breaches": float("nan"), "breach_rate": float("nan")}
        out["backtest"][f"breach_rate_{pct}_param"] = bt_p["breach_rate"]
        out["backtest"][f"n_breaches_{pct}_param"] = bt_p["n_breaches"]
        out["backtest"][f"n_obs_{pct}_param"] = bt_p["n_obs"]

    return out


# --- helpers ---

def _erfinv(x: float) -> float:
    """Approximate inverse error function.

    Uses a rational approximation (Winitzki, 2008). Accuracy is sufficient
    for VaR quantiles.
    """
    # Clip to open interval (-1, 1)
    x = float(np.clip(x, -0.999999, 0.999999))

    a = 0.147  # Winitzki constant
    ln = np.log(1.0 - x * x)
    first = 2.0 / (np.pi * a) + ln / 2.0
    second = ln / a
    return float(np.sign(x) * np.sqrt(np.sqrt(first * first - second) - first))