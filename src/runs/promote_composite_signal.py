

"""Phase 8 helper: promote Phase 7 composite signal into the main daily panel.

This script merges the Phase 7 composite (stored as a separate parquet) into the Phase 3/7
research panel so downstream backtests can reference it via `--signal <name>`.

Typical usage (run from project root):

  python src/runs/promote_composite_signal.py \
    --panel data/processed/crsp_daily_with_lagged_funda_phase3.parquet \
    --composite data/outputs/multi_signal/composite_icw_rank_roll252_H5.parquet \
    --out data/processed/crsp_daily_with_lagged_funda_phase3_plus_composite.parquet \
    --name composite_icw

Notes:
- We do NOT change any portfolio logic here.
- We simply add one new column to the panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_composite_df(path: Path, name: str) -> pd.DataFrame:
    """Load composite parquet and return a tidy DataFrame with columns: permno, date, <name>."""
    c = pd.read_parquet(path)

    # Normalize common formats
    # 1) MultiIndex (date, permno) -> reset
    if isinstance(c.index, pd.MultiIndex):
        idx_names = [n or "" for n in c.index.names]
        if set(idx_names) >= {"date", "permno"}:
            c = c.reset_index()
        else:
            # If unnamed/odd, still reset and try to infer later
            c = c.reset_index()

    # 2) If `date` / `permno` are index (not MultiIndex)
    if "date" not in c.columns and c.index.name == "date":
        c = c.reset_index()
    if "permno" not in c.columns and c.index.name == "permno":
        c = c.reset_index()

    # Require keys
    if "date" not in c.columns or "permno" not in c.columns:
        raise ValueError(
            f"Composite parquet must contain keys 'date' and 'permno' as columns or index. Got columns={list(c.columns)} index={c.index.names}."
        )

    c["date"] = pd.to_datetime(c["date"])

    # Infer value column
    if name in c.columns:
        value_col = name
    elif "composite" in c.columns:
        value_col = "composite"
    else:
        # pick first numeric column that isn't a key
        candidates = [
            col
            for col in c.columns
            if col not in {"date", "permno"} and pd.api.types.is_numeric_dtype(c[col])
        ]
        if not candidates:
            raise ValueError(
                f"Could not infer composite value column. Provide a parquet with a numeric value column. Columns={list(c.columns)}"
            )
        value_col = candidates[0]

    out = c[["permno", "date", value_col]].copy()
    out = out.rename(columns={value_col: name})
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["permno", "date"]).copy()
    out["permno"] = out["permno"].astype(int)

    # Ensure uniqueness to avoid duplication on merge
    dup = out.duplicated(subset=["permno", "date"]).sum()
    if dup:
        raise ValueError(f"Composite has duplicate (permno,date) rows: {dup}")

    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Promote Phase 7 composite signal into main panel")
    parser.add_argument(
        "--panel",
        required=True,
        help="Main panel parquet path (e.g., data/processed/crsp_daily_with_lagged_funda_phase3.parquet)",
    )
    parser.add_argument(
        "--composite",
        required=True,
        help="Composite parquet path (e.g., data/outputs/multi_signal/composite_icw_rank_roll252_H5.parquet)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output parquet path with composite column added",
    )
    parser.add_argument(
        "--name",
        default="composite_icw",
        help="Name of the composite column to add",
    )

    args = parser.parse_args(argv)

    root = _project_root()
    panel_path = Path(args.panel)
    comp_path = Path(args.composite)
    out_path = Path(args.out)

    if not panel_path.is_absolute():
        panel_path = root / panel_path
    if not comp_path.is_absolute():
        comp_path = root / comp_path
    if not out_path.is_absolute():
        out_path = root / out_path

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel parquet not found: {panel_path}")
    if not comp_path.exists():
        raise FileNotFoundError(f"Composite parquet not found: {comp_path}")

    df = pd.read_parquet(panel_path)
    if "date" not in df.columns or "permno" not in df.columns:
        raise KeyError("Panel parquet must contain columns 'permno' and 'date'.")

    df["date"] = pd.to_datetime(df["date"])

    comp = _load_composite_df(comp_path, name=str(args.name))

    # Merge
    out = df.merge(comp, on=["permno", "date"], how="left", validate="one_to_one")

    # Diagnostics
    n_total = len(out)
    n_nonnull = int(out[str(args.name)].notna().sum())
    pct = 100.0 * n_nonnull / max(n_total, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("\n================ PROMOTE COMPOSITE SIGNAL ================")
    print(f"Panel in : {panel_path}")
    print(f"Composite: {comp_path}")
    print(f"Column   : {args.name}")
    print(f"Merged   : {n_nonnull:,} / {n_total:,} rows non-null ({pct:.2f}%)")
    print(f"Wrote    : {out_path}")
    print("==========================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())