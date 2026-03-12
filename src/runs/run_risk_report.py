

"""Run Phase 10 Risk Dashboard.

This script loads realized portfolio returns from a baseline backtest output
folder (period_returns.csv) and produces a Phase 10 risk report bundle via
`src.risk.dashboard`.

Typical usage:
  # Point directly at a period_returns.csv
  python src/runs/run_risk_report.py \
    --period_returns data/outputs/baseline/signal=composite_raw__REBALANCE_FREQUENCY=W__REBALANCE_DAY=FRI__LONG_QUANTILE=0.9__SHORT_QUANTILE=0.1__USE_OPTIMIZER=True/period_returns.csv

  # Or point at the baseline folder containing period_returns.csv
  python src/runs/run_risk_report.py \
    --baseline_dir data/outputs/baseline/signal=composite_raw__REBALANCE_FREQUENCY=W__REBALANCE_DAY=FRI__LONG_QUANTILE=0.9__SHORT_QUANTILE=0.1__USE_OPTIMIZER=True

  # Or let it auto-select the newest baseline run under data/outputs/baseline
  python src/runs/run_risk_report.py

Outputs are written under:
  data/outputs/risk_reports/<label>/

The risk engine is analysis-only and does not modify any backtest outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.risk.dashboard import build_risk_dashboard, load_returns_from_period_returns_csv


DEFAULT_BASELINE_ROOT = Path("data/outputs/baseline")
DEFAULT_RISK_ROOT = Path("data/outputs/risk_reports")


def _find_latest_baseline_run(root: Path) -> Path:
    """Pick the most recently modified baseline run directory that contains period_returns.csv."""
    if not root.exists():
        raise FileNotFoundError(f"Baseline root not found: {root}")

    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "period_returns.csv").exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No baseline run folders with period_returns.csv found under: {root}")

    candidates.sort(key=lambda x: (x / "period_returns.csv").stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_period_returns_path(period_returns: Optional[str], baseline_dir: Optional[str]) -> Path:
    if period_returns:
        p = Path(period_returns)
        if not p.exists():
            raise FileNotFoundError(f"period_returns.csv not found: {p}")
        return p

    if baseline_dir:
        d = Path(baseline_dir)
        p = d / "period_returns.csv"
        if not p.exists():
            raise FileNotFoundError(f"period_returns.csv not found in baseline_dir: {p}")
        return p

    latest_dir = _find_latest_baseline_run(DEFAULT_BASELINE_ROOT)
    return latest_dir / "period_returns.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 10 risk report on baseline returns")
    ap.add_argument(
        "--period_returns",
        type=str,
        default=None,
        help="Path to a period_returns.csv file (preferred explicit input)",
    )
    ap.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Path to a baseline output directory containing period_returns.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for risk report. Default: data/outputs/risk_reports/<label>",
    )
    ap.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for risk report folder (default: baseline folder name)",
    )

    args = ap.parse_args()

    pr_path = _resolve_period_returns_path(args.period_returns, args.baseline_dir)
    baseline_folder = pr_path.parent

    label = args.label or baseline_folder.name

    out_dir = Path(args.out_dir) if args.out_dir else (DEFAULT_RISK_ROOT / label)

    # Load returns
    returns = load_returns_from_period_returns_csv(pr_path)

    # Build dashboard bundle
    build_risk_dashboard(returns, out_dir)

    print("================ PHASE 10 RISK REPORT ================")
    print(f"Input period_returns: {pr_path}")
    print(f"Baseline folder:       {baseline_folder}")
    print(f"Output dir:            {out_dir}")
    print("Wrote: risk_summary.json, var_es.json, factor_exposures.json, stress_tests.csv + plots")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())