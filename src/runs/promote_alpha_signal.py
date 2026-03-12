"""Phase 8 helper: promote one or more alpha module outputs into the main daily panel.

This script computes signal(s) using module(s) in `src/alphas/<alpha>.py` and merges them into
an existing panel parquet on (permno, date). It then writes a new parquet that includes the
additional signal column(s).

Usage (run from project root):

  python src/runs/promote_alpha_signal.py \
    --panel data/processed/crsp_daily_with_lagged_funda_phase3_plus_composite.parquet \
    --alpha beta \
    --out data/processed/crsp_daily_with_lagged_funda_phase3_plus_composite_plus_beta.parquet \
    --name beta_sig

  python src/runs/promote_alpha_signal.py \
    --panel data/processed/crsp_daily_with_lagged_funda_phase3.parquet \
    --alpha beta STR idiosyncratic_vol \
    --name beta_sig STR idiosyncratic_vol \
    --out data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet

Notes:
- Each alpha module must define `compute_signal(df)` or `compute(df)`.
- The returned DataFrame must include columns: permno, date, and either:
  (a) the requested signal column name, or
  (b) exactly one non-key value column, which will be renamed to the requested name.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional

import pandas as pd



def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_alpha_module_name(alpha_name: str) -> str:
    """Map requested signal names to importable module names.

    This lets us store a signal under a different column name than the source module,
    e.g. import `src.alphas.beta` but save the result as `beta_sig`.
    """
    module_aliases = {
        "beta_sig": "beta_sig",
    }
    return module_aliases.get(alpha_name, alpha_name)


def _normalize_signal_output(sig_df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    key_cols = {"permno", "date"}
    if not key_cols.issubset(sig_df.columns):
        raise ValueError(
            f"Alpha output must include key columns ['permno','date']. Got columns={list(sig_df.columns)}"
        )

    if target_name not in sig_df.columns:
        value_cols = [c for c in sig_df.columns if c not in key_cols]
        if len(value_cols) == 1:
            sig_df = sig_df.rename(columns={value_cols[0]: target_name})
        else:
            raise ValueError(
                f"Alpha output must include columns ['permno','date','{target_name}'] or exactly one "
                f"non-key value column. Got columns={list(sig_df.columns)}"
            )

    sig_df = sig_df[["permno", "date", target_name]].copy()
    sig_df["date"] = pd.to_datetime(sig_df["date"])
    sig_df["permno"] = pd.to_numeric(sig_df["permno"], errors="coerce").astype("Int64")
    sig_df = sig_df.dropna(subset=["permno", "date"]).copy()
    sig_df["permno"] = sig_df["permno"].astype(int)

    dup = int(sig_df.duplicated(subset=["permno", "date"]).sum())
    if dup:
        raise ValueError(f"Alpha output has duplicate (permno,date) rows for '{target_name}': {dup}")

    return sig_df


def _promote_one_signal(df: pd.DataFrame, alpha_name: str, col_name: str) -> tuple[pd.DataFrame, int, int, float]:
    # If the requested output column is already in the panel, reuse it directly.
    # This supports fundamental signals like `op` and `be` that already exist in
    # the merged research panel and do not have src/alphas modules.
    if col_name in df.columns:
        n_total = len(df)
        n_nonnull = int(df[col_name].notna().sum())
        pct = 100.0 * n_nonnull / max(n_total, 1)
        return df, n_nonnull, n_total, pct

    try:
        module_name = _resolve_alpha_module_name(alpha_name)
        mod = importlib.import_module(f"src.alphas.{module_name}")
    except Exception as e:
        raise ImportError(f"Could not import src.alphas.{_resolve_alpha_module_name(alpha_name)}") from e

    compute_fn = getattr(mod, "compute_signal", None) or getattr(mod, "compute", None)
    if compute_fn is None:
        raise AttributeError(f"src.alphas.{alpha_name} must define compute_signal(df) or compute(df).")

    sig_df = compute_fn(df)
    if not isinstance(sig_df, pd.DataFrame):
        raise TypeError(f"src.alphas.{alpha_name} compute function must return a pandas DataFrame")

    sig_df = _normalize_signal_output(sig_df=sig_df, target_name=col_name)

    out = df.merge(sig_df, on=["permno", "date"], how="left", validate="one_to_one")

    n_total = len(out)
    n_nonnull = int(out[col_name].notna().sum())
    pct = 100.0 * n_nonnull / max(n_total, 1)
    return out, n_nonnull, n_total, pct


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Promote alpha signal into main panel parquet")
    parser.add_argument("--panel", required=True, help="Input panel parquet")
    parser.add_argument(
        "--alpha",
        required=True,
        nargs="+",
        help="One or more alpha module names under src/alphas (e.g., beta STR idiosyncratic_vol)",
    )
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument(
        "--name",
        nargs="*",
        default=None,
        help="Optional output column names. Must have the same length/order as --alpha if provided.",
    )

    args = parser.parse_args(argv)

    root = _project_root()

    panel_path = Path(args.panel)
    if not panel_path.is_absolute():
        panel_path = root / panel_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel parquet not found: {panel_path}")

    alpha_names = [str(x) for x in args.alpha]
    if args.name is None:
        col_names = list(alpha_names)
    else:
        col_names = [str(x) for x in args.name]
        if len(col_names) != len(alpha_names):
            raise ValueError("If --name is provided, it must have the same number of entries as --alpha.")

    # Ensure repo root is on sys.path so `import src.alphas.*` works when the script
    # is run as `python src/runs/promote_alpha_signal.py` from the project root.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    df = pd.read_parquet(panel_path)
    if "permno" not in df.columns or "date" not in df.columns:
        raise KeyError("Panel parquet must contain columns 'permno' and 'date'.")
    df["date"] = pd.to_datetime(df["date"])

    promotion_stats = []
    out = df.copy()

    for alpha_name, col_name in zip(alpha_names, col_names):
        out, n_nonnull, n_total, pct = _promote_one_signal(out, alpha_name=alpha_name, col_name=col_name)
        promotion_stats.append(
            {
                "alpha_name": alpha_name,
                "col_name": col_name,
                "n_nonnull": n_nonnull,
                "n_total": n_total,
                "pct": pct,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("\n================ PROMOTE ALPHA SIGNAL(S) ================")
    print(f"Panel in : {panel_path}")
    for stat in promotion_stats:
        print(f"Alpha    : src.alphas.{stat['alpha_name']}")
        print(f"Column   : {stat['col_name']}")
        print(f"Merged   : {stat['n_nonnull']:,} / {stat['n_total']:,} rows non-null ({stat['pct']:.2f}%)")
        print("--------------------------------------------------------")
    print(f"Wrote    : {out_path}")
    print("========================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())