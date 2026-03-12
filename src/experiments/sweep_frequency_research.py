import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _safe_tag(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace(" ", "")
        .replace(",", "-")
        .replace(":", "-")
        .replace("=", "-")
    )


def _inc_output_path(
    out_dir: Path,
    roll: int,
    horizon: int,
    signals: list[str],
    rebalance_frequency: str,
    rebalance_day: str,
    orth: bool,
    orth_threshold: float,
) -> Path:
    signals_for_tag = list(signals)
    if orth:
        signals_for_tag = [f"{s}_orth" for s in signals_for_tag]

    signal_tag = _safe_tag("-".join(signals_for_tag))
    rebalance_tag = _safe_tag(f"{rebalance_frequency}-{rebalance_day}")

    tag = f"__SIG={signal_tag}__RB={rebalance_tag}"
    if orth:
        tag += f"__ORTH_thr{orth_threshold:.2f}"

    return out_dir / f"incremental_sharpe_ENGINE_roll{roll}_H{horizon}{tag}.csv"


def _run(cmd: list[str]) -> None:
    print("\nRUNNING:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _full_row(rows: list[dict]) -> dict:
    for r in rows:
        if r["portfolio"] == "composite_full":
            return r
    raise ValueError("composite_full row not found")


def _drop_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["portfolio"].startswith("composite_no_")]


def _parse_float(x: str) -> float:
    if x is None or x == "":
        return float("nan")
    return float(x)


def _signal_from_drop_portfolio(portfolio_name: str) -> str:
    base = portfolio_name.replace("composite_no_", "")
    for suffix in ["_orth", "_cluster_orth"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def _pick_signal_to_drop(rows: list[dict]) -> tuple[str | None, float]:
    """
    Returns:
        (signal_to_drop, sharpe_improvement)
    improvement = drop_sharpe - full_sharpe
    """
    full = _full_row(rows)
    full_sharpe = _parse_float(full["sharpe"])

    best_signal = None
    best_improvement = 0.0

    for r in _drop_rows(rows):
        drop_sharpe = _parse_float(r["sharpe"])
        improvement = drop_sharpe - full_sharpe
        if improvement > best_improvement:
            best_improvement = improvement
            best_signal = _signal_from_drop_portfolio(r["portfolio"])

    return best_signal, best_improvement


def _write_iteration_note(
    run_dir: Path,
    iteration: int,
    signals: list[str],
    inc_csv: Path,
    removed_signal: str | None,
    improvement: float,
) -> None:
    signal_tag = _safe_tag("-".join(signals))
    note = run_dir / f"iteration_{iteration:02d}__SIG={signal_tag}_summary.txt"
    with note.open("w") as f:
        f.write(f"iteration: {iteration}\n")
        f.write(f"signals_used: {signals}\n")
        f.write(f"incremental_csv: {inc_csv}\n")
        f.write(f"removed_signal_next: {removed_signal}\n")
        f.write(f"sharpe_improvement_if_removed: {improvement:.6f}\n")


def _run_one_iteration(
    python_bin: str,
    data: str,
    signals: list[str],
    rebalance_frequency: str,
    rebalance_day: str,
    orth: bool,
    orth_threshold: float,
    roll: int,
    horizon: int,
    base_out_dir: Path,
    iteration: int,
    run_correlation: bool,
    run_ic_weights: bool,
) -> Path:
    run_dir = base_out_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Correlation
    if run_correlation:
        corr_cmd = [
            python_bin,
            "src/runs/run_signal_correlation.py",
            "--data", data,
            "--signals", *signals,
            "--rebalance-frequency", rebalance_frequency,
            "--rebalance-day", rebalance_day,
            "--out-dir", str(run_dir),
        ]
        _run(corr_cmd)

    # 2) IC weights
    if run_ic_weights:
        ic_cmd = [
            python_bin,
            "src/runs/run_IC_weightsavg.py",
            "--data", data,
            "--signals", *signals,
            "--roll", str(roll),
            "--horizon", str(horizon),
            "--rebalance-frequency", rebalance_frequency,
            "--rebalance-day", rebalance_day,
            "--out_dir", str(run_dir),
        ]
        if orth:
            ic_cmd += ["--orth", "--orth_threshold", str(orth_threshold)]
        _run(ic_cmd)

    # 3) Incremental Sharpe
    inc_cmd = [
        python_bin,
        "src/runs/run_incremental_sharpe.py",
        "--data", data,
        "--signals", *signals,
        "--roll", str(roll),
        "--horizon", str(horizon),
        "--rebalance-frequency", rebalance_frequency,
        "--rebalance-day", rebalance_day,
        "--out_dir", str(run_dir),
    ]
    if orth:
        inc_cmd += ["--orth", "--orth_threshold", str(orth_threshold)]
    _run(inc_cmd)

    inc_path = _inc_output_path(
        out_dir=run_dir,
        roll=roll,
        horizon=horizon,
        signals=signals,
        rebalance_frequency=rebalance_frequency,
        rebalance_day=rebalance_day,
        orth=orth,
        orth_threshold=orth_threshold,
    )
    if not inc_path.exists():
        raise FileNotFoundError(f"Expected incremental output not found: {inc_path}")

    return inc_path


def main():
    parser = argparse.ArgumentParser(
        description="Sweep rebalance frequencies for raw and orth research, with greedy backward pruning."
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Initial candidate signals from alpha validation.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--roll", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--frequencies",
        nargs="+",
        default=["W", "BW", "TW", "M"],
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
    )
    parser.add_argument("--rebalance-day", default="FRI", choices=["MON", "TUE", "WED", "THU", "FRI"])
    parser.add_argument("--orth-threshold", type=float, default=0.30)
    parser.add_argument("--min-signals", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--out-dir", default="data/outputs/frequency_research_sweep")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run correlation and IC weights only on the first iteration for each frequency/mode; later iterations run incremental Sharpe only.",
    )
    args = parser.parse_args()

    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    modes = [
        ("raw", False),
        (f"orth_thr{args.orth_threshold:.2f}", True),
    ]

    for freq in args.frequencies:
        for mode_name, orth in modes:
            mode_dir = root_out / freq / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)

            current_signals = list(args.signals)
            iteration = 0
            last_rows = None
            last_inc_path = None

            print("\n" + "=" * 100)
            print(f"START | frequency={freq} | mode={mode_name}")
            print("=" * 100)

            while True:
                if len(current_signals) < args.min_signals:
                    print(f"Stopping: reached min_signals={args.min_signals}")
                    break

                iteration += 1
                run_correlation = (iteration == 1) or (not args.fast_mode)
                run_ic_weights = (iteration == 1) or (not args.fast_mode)

                inc_path = _run_one_iteration(
                    python_bin=args.python_bin,
                    data=args.data,
                    signals=current_signals,
                    rebalance_frequency=freq,
                    rebalance_day=args.rebalance_day,
                    orth=orth,
                    orth_threshold=args.orth_threshold,
                    roll=args.roll,
                    horizon=args.horizon,
                    base_out_dir=mode_dir,
                    iteration=iteration,
                    run_correlation=run_correlation,
                    run_ic_weights=run_ic_weights,
                )

                rows = _read_csv(inc_path)
                signal_to_drop, improvement = _pick_signal_to_drop(rows)

                _write_iteration_note(
                    run_dir=mode_dir,
                    iteration=iteration,
                    signals=current_signals,
                    inc_csv=inc_path,
                    removed_signal=signal_to_drop,
                    improvement=improvement,
                )

                last_rows = rows
                last_inc_path = inc_path

                if signal_to_drop is None or improvement <= args.min_improvement:
                    print("No further Sharpe improvement from dropping signals.")
                    break

                if len(current_signals) - 1 < args.min_signals:
                    print(
                        f"Best removable signal is {signal_to_drop}, but dropping it would "
                        f"violate min_signals={args.min_signals}. Stopping here."
                    )
                    break

                print(
                    f"Drop next signal: {signal_to_drop} | "
                    f"Sharpe improvement={improvement:.6f}"
                )
                current_signals.remove(signal_to_drop)

            if last_rows is None:
                continue

            full = _full_row(last_rows)
            summary_rows.append({
                "frequency": freq,
                "mode": mode_name,
                "signals_final": ",".join(current_signals),
                "n_signals_final": len(current_signals),
                "sharpe": full.get("sharpe", ""),
                "ann_ret": full.get("ann_ret", ""),
                "ann_vol": full.get("ann_vol", ""),
                "max_drawdown": full.get("max_drawdown", ""),
                "avg_turnover": full.get("avg_turnover", ""),
                "last_incremental_csv": str(last_inc_path),
            })

    summary_path = root_out / "summary_best_composites.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frequency",
                "mode",
                "signals_final",
                "n_signals_final",
                "sharpe",
                "ann_ret",
                "ann_vol",
                "max_drawdown",
                "avg_turnover",
                "last_incremental_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 100)
    print(f"DONE. Summary written to: {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()