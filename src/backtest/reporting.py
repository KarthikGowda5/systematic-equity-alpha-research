"""Reporting utilities (Phase 4 baseline).

This module writes the baseline backtest report artifacts:
- period_returns.csv
- turnover.csv
- equity_curve.csv
- metrics.json
- settings_used.json
- equity_curve.png

No tables beyond these essentials; keep it phase-appropriate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def write_report(
    out_dir: Path,
    period_returns: pd.Series,
    turnover: pd.Series,
    equity_curve: pd.Series,
    metrics: Dict[str, float],
    settings_used: Dict[str, Any],
    weights: Optional[pd.DataFrame] = None,
    period_details: Optional[pd.DataFrame] = None,
    plot_title: str = "Baseline Equity Curve",
) -> None:
    """Write baseline report artifacts to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save series
    period_returns.to_csv(out_dir / "period_returns.csv", index=True)
    turnover.to_csv(out_dir / "turnover.csv", index=True)
    equity_curve.to_csv(out_dir / "equity_curve.csv", index=True)
    # Save weights (if provided)
    if weights is not None and not weights.empty:
        weights.to_csv(out_dir / "weights.csv", index=False)
    # Save detailed per-period execution + cost breakdown (Phase 11 support)
    if period_details is not None and not period_details.empty:
        period_details.to_csv(out_dir / "period_details.csv", index=False)

    # Save metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save settings snapshot
    with open(out_dir / "settings_used.json", "w", encoding="utf-8") as f:
        json.dump(settings_used, f, indent=2, default=str)

    # Equity curve plot
    _write_equity_curve_plot(
        out_path=out_dir / "equity_curve.png",
        equity_curve=equity_curve,
        title=plot_title,
    )


def _write_equity_curve_plot(out_path: Path, equity_curve: pd.Series, title: str) -> None:
    """Write equity curve PNG. Plot is helpful but not required for correctness."""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(equity_curve.index, equity_curve.values)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        # Save plot error for debugging without failing the run
        with open(out_path.with_suffix(".plot_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))