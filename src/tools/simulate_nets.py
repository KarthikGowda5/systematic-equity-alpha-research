

"""Simulate net return series from existing backtest outputs.

Uses:
- Gross run: period_returns.csv + turnover.csv (costs OFF)
- Model-cost run: period_returns.csv (costs ON, your Phase 9 model)

Outputs a single CSV with gross + multiple net series + equity curves, plus prints a small summary.

Example:
  python src/tools/simulate_nets.py \
    --gross_dir data/outputs/baseline/signal=composite_raw__...__h=XXXX \
    --model_dir data/outputs/baseline/signal=composite_raw__...__h=YYYY \
    --out data/outputs/baseline/net_sims__XXXX__YYYY.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)

    # Many outputs are written with date as the unnamed first column (pandas index).
    if "date" not in df.columns and len(df.columns) > 0:
        first_col = df.columns[0]

        # Common pandas index column names.
        if first_col in ("", "Unnamed: 0") or str(first_col).lower().startswith("unnamed"):
            df = df.rename(columns={first_col: "date"})
        else:
            # If the first column looks like dates, treat it as date.
            sample = df[first_col].astype(str).head(5)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= 3:
                df = df.rename(columns={first_col: "date"})

    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a 'date' column (or an unnamed first column with dates)")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    df = df.sort_values("date")
    return df


def _infer_return_col(df: pd.DataFrame) -> str:
    for c in ["portfolio_return", "return", "ret", "period_return"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find return column. Columns={list(df.columns)}")


def _infer_turnover_col(df: pd.DataFrame) -> str:
    for c in ["turnover", "period_turnover", "to"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find turnover column. Columns={list(df.columns)}")


def _ann_sharpe(r: pd.Series, periods_per_year: float) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=0))
    if sd == 0.0:
        return float("nan")
    return (mu / sd) * float(np.sqrt(periods_per_year))


def _max_drawdown(eq: pd.Series) -> float:
    # eq is an equity curve (starts around 1)
    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0
    return float(dd.min())


def simulate(
    gross_dir: Path,
    model_dir: Path,
    out_path: Path,
    bps_list: list[float],
    periods_per_year: float,
) -> pd.DataFrame:
    # Load gross series
    gross_pr = _read_csv(gross_dir / "period_returns.csv")
    gross_to = _read_csv(gross_dir / "turnover.csv")

    gross_ret_col = _infer_return_col(gross_pr)
    to_col = _infer_turnover_col(gross_to)

    gross = gross_pr[["date", gross_ret_col]].rename(columns={gross_ret_col: "gross"})
    to = gross_to[["date", to_col]].rename(columns={to_col: "turnover"})

    # Load model-net series
    model_pr = _read_csv(model_dir / "period_returns.csv")
    model_ret_col = _infer_return_col(model_pr)
    model_net = model_pr[["date", model_ret_col]].rename(columns={model_ret_col: "net_model"})

    # Align
    df = gross.merge(to, on="date", how="inner").merge(model_net, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # Net at fixed bps per $ traded (turnover * cost)
    for bps in bps_list:
        c = float(bps) / 1e4  # bps -> decimal
        name = f"net_{int(bps)}bps"
        df[name] = df["gross"] - c * df["turnover"]

    # Infer the model cost series (in return space)
    df["model_cost"] = df["gross"] - df["net_model"]

    # Equity curves
    for col in ["gross", "net_model"] + [f"net_{int(b)}bps" for b in bps_list]:
        df[f"eq_{col}"] = (1.0 + df[col].astype(float)).cumprod()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary print
    def summary(name: str, r: pd.Series) -> dict:
        eq = (1.0 + r.astype(float)).cumprod()
        return {
            "series": name,
            "periods": int(r.shape[0]),
            "mean_period": float(r.mean()),
            "ann_vol": float(r.std(ddof=0) * np.sqrt(periods_per_year)),
            "sharpe": float(_ann_sharpe(r, periods_per_year)),
            "max_drawdown": float(_max_drawdown(eq)),
        }

    rows = [summary("gross", df["gross"].astype(float))]
    for bps in bps_list:
        rows.append(summary(f"net_{int(bps)}bps", df[f"net_{int(bps)}bps"].astype(float)))
    rows.append(summary("net_model", df["net_model"].astype(float)))

    summ = pd.DataFrame(rows)
    # Append turnover stats (same for all series)
    summ["avg_turnover"] = float(df["turnover"].mean())

    print("\n==== NET RETURN SIMULATION SUMMARY ====")
    print(summ.to_string(index=False))
    print(f"\nWrote: {out_path}")

    return df


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Simulate net returns from gross + model-cost runs")
    p.add_argument(
        "--gross_dir",
        required=True,
        help="Directory containing gross run outputs (period_returns.csv, turnover.csv)",
    )
    p.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing model-cost run outputs (period_returns.csv)",
    )
    p.add_argument(
        "--out",
        default="net_sims.csv",
        help="Output CSV path (default: net_sims.csv)",
    )
    p.add_argument(
        "--bps",
        default="10,20",
        help="Comma-separated list of bps costs per $ traded (default: 10,20)",
    )
    p.add_argument(
        "--periods_per_year",
        type=float,
        default=50.5,
        help="Periods per year for annualization (weekly default: 50.5)",
    )

    args = p.parse_args(argv)

    gross_dir = Path(args.gross_dir)
    model_dir = Path(args.model_dir)
    out_path = Path(args.out)

    bps_list = [float(x.strip()) for x in str(args.bps).split(",") if x.strip()]
    if not bps_list:
        raise ValueError("--bps produced an empty list")

    simulate(
        gross_dir=gross_dir,
        model_dir=model_dir,
        out_path=out_path,
        bps_list=bps_list,
        periods_per_year=float(args.periods_per_year),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())