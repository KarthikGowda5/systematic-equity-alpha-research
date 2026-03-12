"""Phase 8 diagnostic: analyze optimized portfolio exposures from saved weights.

This script reads `weights.csv` produced by the baseline backtest runner and computes
per-rebalance-date exposure diagnostics:

- Portfolio beta exposure: sum_i w_i * beta_i
- Sector exposures (if sector column exists): sum_i w_i by sector
- Concentration: effective number of positions N_eff = 1 / sum_i w_i^2
- Gross/net checks

Usage (run from project root):

  python src/runs/analyse_optimizer_exposures.py \
    --run-dir data/outputs/baseline/<your_run_id_folder> \
    --panel data/processed/crsp_daily_with_lagged_funda_phase3_plus_composite_plus_beta.parquet \
    --beta-col beta \
    --sector-col sector

Outputs written into --run-dir by default:
- exposure_summary.csv
- beta_exposure_series.csv
- sector_exposure_series.csv (if available)

Notes:
- Sector analysis requires a sector identifier column present in the panel. If you don't have one yet,
  the script will skip sector outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_weights(run_dir: Path) -> pd.DataFrame:
    w_path = run_dir / "weights.csv"
    if not w_path.exists():
        raise FileNotFoundError(f"weights.csv not found in run-dir: {w_path}")

    w = pd.read_csv(w_path)
    required = {"date", "permno", "weight"}
    missing = required.difference(w.columns)
    if missing:
        raise KeyError(f"weights.csv missing columns: {sorted(missing)}")

    w["date"] = pd.to_datetime(w["date"])
    w["permno"] = pd.to_numeric(w["permno"], errors="coerce").astype("Int64")
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    w = w.dropna(subset=["date", "permno", "weight"]).copy()
    w["permno"] = w["permno"].astype(int)

    # ensure no duplicates
    dup = int(w.duplicated(subset=["date", "permno"]).sum())
    if dup:
        raise ValueError(f"weights.csv has duplicate (date,permno) rows: {dup}")

    return w


def _load_panel(panel_path: Path, beta_col: str, sector_col: str) -> pd.DataFrame:
    df = pd.read_parquet(panel_path)
    if "date" not in df.columns or "permno" not in df.columns:
        raise KeyError("Panel parquet must contain columns 'date' and 'permno'.")

    cols = ["date", "permno"]
    if beta_col in df.columns:
        cols.append(beta_col)
    if sector_col in df.columns:
        cols.append(sector_col)

    x = df[cols].copy()
    x["date"] = pd.to_datetime(x["date"])
    x["permno"] = pd.to_numeric(x["permno"], errors="coerce").astype("Int64")
    x = x.dropna(subset=["date", "permno"]).copy()
    x["permno"] = x["permno"].astype(int)

    return x


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze optimized portfolio exposures from weights.csv")
    parser.add_argument("--run-dir", required=True, help="Run output dir containing weights.csv")
    parser.add_argument(
        "--panel",
        default="data/processed/crsp_daily_with_lagged_funda_phase3_plus_composite_plus_beta.parquet",
        help="Panel parquet used to merge exposures",
    )
    parser.add_argument("--beta-col", default="beta", help="Beta column name in panel")
    parser.add_argument("--sector-col", default="sector", help="Sector column name in panel (optional)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for exposure artifacts (defaults to --run-dir)",
    )

    args = parser.parse_args(argv)

    root = _project_root()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    panel_path = Path(args.panel)
    if not panel_path.is_absolute():
        panel_path = root / panel_path

    out_dir = Path(args.out_dir) if args.out_dir is not None else run_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    w = _load_weights(run_dir)
    panel = _load_panel(panel_path, beta_col=str(args.beta_col), sector_col=str(args.sector_col))

    beta_col = str(args.beta_col)
    sector_col = str(args.sector_col)

    merged = w.merge(panel, on=["date", "permno"], how="left", validate="one_to_one")

    # --- Core diagnostics per date ---
    g = merged.groupby("date", sort=True)

    net = g["weight"].sum().rename("net")
    gross = g["weight"].apply(lambda s: float(np.abs(s).sum())).rename("gross")
    max_abs_w = g["weight"].apply(lambda s: float(np.max(np.abs(s.values))) if len(s) else np.nan).rename("max_abs_w")

    # effective number of positions: 1 / sum w^2
    neff = g["weight"].apply(lambda s: float(1.0 / np.sum(np.square(s.values))) if np.sum(np.square(s.values)) > 0 else np.nan).rename("n_eff")

    # beta exposure if available
    if beta_col in merged.columns:
        beta_exposure = g.apply(lambda x: float(np.nansum(x["weight"].values * x[beta_col].values))).rename("beta_exposure")
    else:
        beta_exposure = pd.Series(index=net.index, dtype=float, name="beta_exposure")

    summary = pd.concat([net, gross, max_abs_w, neff, beta_exposure], axis=1).reset_index()

    beta_path = out_dir / "beta_exposure_series.csv"
    summary[["date", "beta_exposure"]].to_csv(beta_path, index=False)

    # --- Sector exposures (optional) ---
    sector_written = False
    if sector_col in merged.columns:
        # keep only rows with sector
        tmp = merged.dropna(subset=[sector_col]).copy()
        if not tmp.empty:
            sector_tbl = (
                tmp.groupby(["date", sector_col])["weight"].sum().reset_index().rename(columns={"weight": "sector_weight"})
            )
            sector_tbl.to_csv(out_dir / "sector_exposure_series.csv", index=False)
            sector_written = True

            # Compute per-date max absolute sector exposure
            sector_max = (
                sector_tbl.assign(abs_w=lambda d: np.abs(d["sector_weight"]))
                .groupby("date")["abs_w"]
                .max()
                .rename("sector_max_abs")
            )

            # Merge into summary for reporting
            summary = summary.merge(sector_max.reset_index(), on="date", how="left")

    summary_path = out_dir / "exposure_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --- Print concise console summary ---
    def _fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "nan"
        return f"{x:.6f}"

    print("\n================ OPTIMIZER EXPOSURE DIAGNOSTICS ================")
    print(f"Run dir : {run_dir}")
    print(f"Panel   : {panel_path}")
    print(f"W rows  : {len(w):,}")
    print(f"Dates   : {summary['date'].nunique():,}")

    if len(summary) > 0:
        print("\n---- Net / Gross ----")
        print(f"net  mean: {_fmt(float(summary['net'].mean()))}   max|net|: {_fmt(float(np.max(np.abs(summary['net'].values))))}")
        print(f"gross mean: {_fmt(float(summary['gross'].mean()))}  min/max: {_fmt(float(summary['gross'].min()))} / {_fmt(float(summary['gross'].max()))}")

        print("\n---- Concentration ----")
        print(f"max|w| mean: {_fmt(float(summary['max_abs_w'].mean()))}  max: {_fmt(float(summary['max_abs_w'].max()))}")
        print(f"N_eff   mean: {_fmt(float(summary['n_eff'].mean()))}  min/max: {_fmt(float(summary['n_eff'].min()))} / {_fmt(float(summary['n_eff'].max()))}")

        if beta_col in merged.columns:
            print("\n---- Beta exposure ----")
            print(f"beta_exposure mean: {_fmt(float(summary['beta_exposure'].mean()))}")
            print(f"beta_exposure std : {_fmt(float(summary['beta_exposure'].std(ddof=0)))}")
            print(f"max|beta_exposure|: {_fmt(float(np.max(np.abs(summary['beta_exposure'].values))))}")
        else:
            print("\n---- Beta exposure ----")
            print(f"beta column '{beta_col}' not found in merged data; skipping.")

        if sector_written:
            print("\n---- Sector exposure ----")
            print(f"sector_max_abs mean: {_fmt(float(summary['sector_max_abs'].mean()))}")
            print(f"sector_max_abs std : {_fmt(float(summary['sector_max_abs'].std(ddof=0)))}")
            print(f"max|sector_exposure|: {_fmt(float(np.max(summary['sector_max_abs'].values)))}")
            print("Wrote sector_exposure_series.csv")
        else:
            print("\n---- Sector exposure ----")
            print(f"sector column '{sector_col}' not found or empty; skipping.")

    print("\nWrote:")
    print(f"- {summary_path}")
    print(f"- {beta_path}")
    if sector_written:
        print(f"- {out_dir / 'sector_exposure_series.csv'}")
    print("===============================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())