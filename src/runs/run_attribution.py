

from __future__ import annotations

import argparse
from pathlib import Path

from src.attribution.performance_attribution import AttributionInputs, run_attribution, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 11 performance attribution.")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the baseline backtest run directory containing weights.csv, period_returns.csv, turnover.csv, metrics.json, and settings_used.json.",
    )
    parser.add_argument(
        "--panel",
        required=True,
        help="Path to the processed panel parquet/csv used for the attributed run.",
    )
    parser.add_argument(
        "--factors",
        required=True,
        help="Path to the factor file (for example ff_factors_mom_daily.csv).",
    )
    parser.add_argument(
        "--signal_weights",
        default=None,
        help="Optional path to daily multi-signal combination weights for sleeve attribution.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Root output directory for attribution artifacts.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional explicit output folder name. If omitted, the run directory name is used.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    panel_path = Path(args.panel).expanduser().resolve()
    factor_path = Path(args.factors).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    signal_weights_path = (
        Path(args.signal_weights).expanduser().resolve() if args.signal_weights is not None else None
    )

    out_name = args.name if args.name else run_dir.name
    final_out_dir = out_root / out_name

    inputs = AttributionInputs(
        run_dir=run_dir,
        panel_path=panel_path,
        factor_path=factor_path,
        signal_weights_path=signal_weights_path,
    )

    results = run_attribution(inputs)
    write_outputs(results, final_out_dir)

    print("================ PHASE 11 ATTRIBUTION ================")
    print(f"Run dir   : {run_dir}")
    print(f"Panel     : {panel_path}")
    print(f"Factors   : {factor_path}")
    print(f"Out dir   : {final_out_dir}")
    if signal_weights_path is not None:
        print(f"Sig wts   : {signal_weights_path}")
    print("")
    print("Wrote:")
    print(f"- {final_out_dir / 'attribution_daily.csv'}")
    print(f"- {final_out_dir / 'factor_attribution.csv'}")
    print(f"- {final_out_dir / 'sector_attribution.csv'}")
    print(f"- {final_out_dir / 'cost_attribution.csv'}")
    print(f"- {final_out_dir / 'sleeve_attribution.csv'}")
    print(f"- {final_out_dir / 'attribution_summary.json'}")
    print(f"- {final_out_dir / 'attribution_memo.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())