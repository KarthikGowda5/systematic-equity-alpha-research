import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Ensure project root and `src/` are on PYTHONPATH when running as:
# `python src/runs/run_signal_correlation.py`
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root (flagship/)
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtest.universe import UniverseSpec, build_liquidity_universe_mask, apply_universe_mask
from backtest.engine import get_rebalance_dates


MODULE_ALIASES = {
    "beta_sig": "beta_sig",
}


def _resolve_module_name(signal_name: str) -> str:
    return MODULE_ALIASES.get(signal_name, signal_name)


def _normalize_signal_output(out: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """Normalize alpha output to exactly (permno, date, sig_col)."""
    if not isinstance(out, pd.DataFrame):
        raise TypeError("Alpha compute function must return a pandas DataFrame")

    key_cols = {"permno", "date"}
    if not key_cols.issubset(out.columns):
        raise KeyError(
            f"Alpha output must include key columns ['permno','date']. Got columns: {list(out.columns)[:50]}"
        )

    if sig_col not in out.columns:
        value_cols = [c for c in out.columns if c not in key_cols]
        if len(value_cols) == 1:
            out = out.rename(columns={value_cols[0]: sig_col})
        else:
            raise KeyError(
                f"Alpha output must include columns ['permno','date','{sig_col}'] or exactly one "
                f"non-key value column. Got columns: {list(out.columns)[:50]}"
            )

    out = out[["permno", "date", sig_col]].copy()
    out["date"] = pd.to_datetime(out["date"])
    dup = int(out.duplicated(subset=["permno", "date"]).sum())
    if dup:
        raise ValueError(f"Alpha output has duplicate (permno,date) rows for '{sig_col}': {dup}")
    return out


def _merge_signal(base: pd.DataFrame, out: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """Merge a single signal column produced by an alpha module back into the base panel."""
    out = _normalize_signal_output(out, sig_col)

    if sig_col in base.columns:
        base = base.drop(columns=[sig_col])

    return base.merge(out, on=["permno", "date"], how="left", validate="one_to_one")


def _compute_missing_signal(panel: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """Compute a signal from src/alphas only if it is missing from the panel."""
    module_name = _resolve_module_name(sig_col)

    try:
        mod = importlib.import_module(f"alphas.{module_name}")
    except Exception as e:
        raise ImportError(f"Could not import alphas.{module_name} for signal '{sig_col}'") from e

    compute_fn = getattr(mod, "compute_signal", None) or getattr(mod, "compute", None)
    if compute_fn is None:
        # Support older function names used in some modules
        legacy_fn_names = {
            "beta_sig": "compute_beta",
            "idiosyncratic_vol": "compute_idiosyncratic_vol",
            "Residual_mom": "compute_residual_mom",
            "STR": "compute_STR",
        }
        legacy_name = legacy_fn_names.get(sig_col)
        if legacy_name is not None:
            compute_fn = getattr(mod, legacy_name, None)

    if compute_fn is None:
        raise AttributeError(
            f"Alpha module alphas.{module_name} must define compute_signal(), compute(), "
            f"or a supported legacy compute function."
        )

    work = panel.copy()
    if ("ret" not in work.columns) and ("ret_total" in work.columns):
        work["ret"] = work["ret_total"]

    out = compute_fn(work)
    return _normalize_signal_output(out, sig_col)


def prepare_signals(panel: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Ensure each requested signal exists in the panel, computing only missing ones."""
    prepared = panel.copy()
    resolved_signals: list[str] = []

    for sig in signals:
        if sig in prepared.columns:
            resolved_signals.append(sig)
            continue

        sig_out = _compute_missing_signal(prepared, sig)
        prepared = _merge_signal(prepared, sig_out, sig)
        resolved_signals.append(sig)

    return prepared, resolved_signals


def compute_mean_rank_corr_matrix(panel: pd.DataFrame, signals: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Time-average of same-date cross-sectional correlations using cross-sectional pct ranks."""
    panel = panel.dropna(subset=signals).copy()

    for s in signals:
        panel[s + "_rank"] = panel.groupby("date")[s].rank(pct=True)

    rank_cols = [s + "_rank" for s in signals]

    corr_by_date = panel.groupby("date")[rank_cols].corr()

    corr_stack = corr_by_date.stack().reset_index()
    corr_stack.columns = ["date", "signal_1", "signal_2", "corr"]

    mean_corr = corr_stack.groupby(["signal_1", "signal_2"])["corr"].mean().unstack()
    mean_corr.index.name = "signal_1"
    mean_corr.columns.name = "signal_2"

    return mean_corr, rank_cols


def main():
    parser = argparse.ArgumentParser(description="Compute cross-sectional signal correlation matrix.")
    parser.add_argument(
        "--data",
        default="data/processed/crsp_daily_with_lagged_funda_phase3.parquet",
        help="Input panel parquet",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        help="Signal columns to use. Existing panel columns are used directly; missing ones are computed.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/outputs/multi_signal",
        help="Directory to save correlation outputs",
    )
    parser.add_argument(
        "--rebalance-frequency",
        default="W",
        choices=["D", "W", "BW", "TW", "M", "BIWEEKLY", "TRIWEEKLY"],
        help="Rebalance frequency used to select the research grid.",
    )
    parser.add_argument(
        "--rebalance-day",
        default="FRI",
        choices=["MON", "TUE", "WED", "THU", "FRI"],
        help="Weekday used when rebalance frequency is W/BW/TW.",
    )
    args = parser.parse_args()

    panel = pd.read_parquet(args.data)

    if not np.issubdtype(panel["date"].dtype, np.datetime64):
        panel["date"] = pd.to_datetime(panel["date"])

    panel, signals = prepare_signals(panel, args.signals)

    rebalance_dates = get_rebalance_dates(
        pd.DatetimeIndex(sorted(panel["date"].dropna().unique())),
        {
            "REBALANCE_FREQUENCY": str(args.rebalance_frequency),
            "REBALANCE_DAY": str(args.rebalance_day),
        },
    )
    panel = panel[panel["date"].isin(rebalance_dates)].copy()

    univ_spec = UniverseSpec(filter_type="ADV", top_n=1000, recon_freq="M", adv_window=20)
    mask = build_liquidity_universe_mask(panel, univ_spec)
    panel = apply_universe_mask(panel, mask)

    mean_corr, rank_cols = compute_mean_rank_corr_matrix(panel, signals)

    print(mean_corr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_corr.to_csv(out_dir / "mean_rank_corr_matrix.csv")

    meta = {
        "data": str(args.data),
        "signals": signals,
        "rank_cols": rank_cols,
        "rebalance_frequency": str(args.rebalance_frequency),
        "rebalance_day": args.rebalance_day,
        "universe": {
            "type": "ADV",
            "top_n": 1000,
            "recon_freq": "M",
            "adv_window": 20,
            "source": "backtest.universe",
        },
    }

    with open(out_dir / "mean_rank_corr_matrix.meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()