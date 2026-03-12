"""Risk dashboard assembly (Phase 10).

This module orchestrates Phase 10 risk components and writes a simple
file-based report bundle (JSON/CSV/PNG) under an output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import compute_standard_metrics, rolling_vol, rolling_sharpe
from .var_es import compute_var_es
from .stress_test import compute_stress_tests
from .factor_regression import FactorRegressionConfig, compute_factor_exposures, load_factor_csv


def _to_series(returns: pd.Series) -> pd.Series:
    r = returns.copy()
    if not isinstance(r.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        r.index = pd.to_datetime(r.index)
    r = pd.to_numeric(r, errors="coerce").astype(float).dropna().sort_index()
    return r


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _plot_series(path: Path, s: pd.Series, title: str, ylabel: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_hist(path: Path, r: pd.Series):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(r.values, bins=60)
    ax.set_title("Return Distribution")
    ax.set_xlabel("return")
    ax.set_ylabel("count")
    ax.grid(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def build_risk_dashboard(
    returns: pd.Series,
    out_dir: str | Path,
):
    """Run full Phase 10 risk analysis and write dashboard outputs."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    r = _to_series(returns)

    # Standard metrics
    metrics, equity, drawdown = compute_standard_metrics(r)
    _write_json(out_path / "risk_summary.json", metrics)

    # VaR / ES
    var_es = compute_var_es(r)
    _write_json(out_path / "var_es.json", var_es)

    # Stress tests
    stress = compute_stress_tests(r)
    stress.to_csv(out_path / "stress_tests.csv")

    # Factor exposures (may fail if factor file missing)
    factor_df = None
    try:
        factors = compute_factor_exposures(r)
        _write_json(out_path / "factor_exposures.json", factors)

        # Load standardized factor time series for additional diagnostics/plots
        cfg = FactorRegressionConfig()
        factor_df = load_factor_csv(cfg.factors_csv, cfg.date_col)
    except Exception as e:
        _write_json(out_path / "factor_exposures.json", {"error": str(e)})

    # Plots
    _plot_series(out_path / "equity_curve.png", equity, "Equity Curve", "equity")
    _plot_series(out_path / "drawdown.png", drawdown, "Drawdown", "drawdown")
    _plot_hist(out_path / "return_hist.png", r)

    rv = rolling_vol(r)
    _plot_series(out_path / "rolling_vol.png", rv.dropna(), "Rolling Vol", "vol")

    rs = rolling_sharpe(r)
    _plot_series(out_path / "rolling_sharpe.png", rs.dropna(), "Rolling Sharpe", "sharpe")

    # Rolling market beta (institutional check for time-varying directionality)
    # beta_t = Cov(ret, mkt) / Var(mkt)
    if factor_df is not None and "mkt" in factor_df.columns:
        try:
            merged = pd.concat([r.rename("ret"), factor_df["mkt"].rename("mkt")], axis=1).dropna()
            window = 60
            cov = merged["ret"].rolling(window).cov(merged["mkt"])
            var = merged["mkt"].rolling(window).var()
            beta = cov / var
            beta = beta.replace([np.inf, -np.inf], np.nan)
            _plot_series(
                out_path / "rolling_beta_mkt.png",
                beta.dropna(),
                "Rolling Market Beta (60d)",
                "beta",
            )
        except Exception as e:
            # Don't fail report generation due to optional diagnostic
            _write_json(out_path / "rolling_beta_mkt_error.json", {"error": str(e)})

    return {
        "metrics": metrics,
        "var_es": var_es,
        "stress": stress,
    }


def load_returns_from_period_returns_csv(path: str | Path) -> pd.Series:
    """Load returns from baseline backtest period_returns.csv."""

    df = pd.read_csv(path)

    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    for c in ["ret", "return", "portfolio_return", "net_ret", "gross_ret"]:
        if c in df.columns:
            ret_col = c
            break
    else:
        raise ValueError("Return column not found in period_returns.csv")

    s = pd.to_numeric(df[ret_col], errors="coerce")
    s.index = df[date_col]
    s.name = "ret"

    return _to_series(s)