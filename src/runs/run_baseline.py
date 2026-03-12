"""Phase 4 — Baseline Backtest Runner (CLI override driven)

Usage examples (run from project root):

  # Weekly (Friday) rebalance, signal=op
  python -m src.runs.run_baseline --signal op --set REBALANCE_FREQUENCY=W --set REBALANCE_DAY=FRI

  # Daily rebalance, signal=inv
  python -m src.runs.run_baseline --signal inv --set REBALANCE_FREQUENCY=D

  # Monthly (EOM) rebalance, signal=be
  python -m src.runs.run_baseline --signal be --set REBALANCE_FREQUENCY=M

  # Weekly (Friday) rebalance with bid-ask spread costs enabled
  python -m src.runs.run_baseline --signal op --set APPLY_SPREAD_COST=true --set SPREAD_CAP=0.20

  # Override any config value (type-cast based on config.py default)
  python -m src.runs.run_baseline --signal op --set LONG_QUANTILE=0.9 --set SHORT_QUANTILE=0.1

This script:
- Loads defaults from your project-level config.py
- Applies repeatable CLI overrides via --set KEY=VALUE
- Runs baseline long/short quantile portfolio construction and holding-period returns
- Writes a baseline backtest report to data/outputs/baseline/<compact_experiment_name>/

Notes:
- Weights are formed at rebalance date t and applied to returns over (t, t_next].
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# -------------------------------------------------------------------
# Allow running as a script: `python src/runs/run_baseline.py ...`
# by ensuring the project root is on sys.path so `import src.*` works.
# -------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.engine import run_baseline_backtest
from src.backtest.reporting import write_report
from src.backtest.universe import UniverseSpec, build_liquidity_universe_mask


# ----------------------------
# Config loading + overrides
# ----------------------------


def _project_root_from_this_file() -> Path:
    # .../flagship/src/runs/run_baseline.py -> .../flagship
    return Path(__file__).resolve().parents[2]


def _load_project_config_module() -> Any:
    """Import project-level config.py.

    We assume you have a `config.py` at the project root, or under `src/config.py`.
    We'll try root first, then src.
    """
    root = _project_root_from_this_file()
    sys.path.insert(0, str(root))

    try:
        return importlib.import_module("config")
    except Exception:
        pass

    try:
        return importlib.import_module("src.config")
    except Exception as e:
        raise ImportError(
            "Could not import config. Expected config.py at project root (flagship/config.py) "
            "or at flagship/src/config.py"
        ) from e


def _extract_uppercase_settings(cfg_mod: Any) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    for k in dir(cfg_mod):
        if k.isupper() and not k.startswith("__"):
            settings[k] = getattr(cfg_mod, k)
    return settings


def _parse_bool(s: str) -> bool:
    s2 = s.strip().lower()
    if s2 in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s2 in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from: {s}")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _cast_value_like(existing: Any, raw: str) -> Any:
    """Cast raw string to the type of `existing` (based on config.py default)."""
    raw = raw.strip()

    if isinstance(existing, bool):
        return _parse_bool(raw)
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(raw)
    if isinstance(existing, float):
        return float(raw)
    if isinstance(existing, str):
        return raw
    if isinstance(existing, (list, tuple)):
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        if len(existing) == 0:
            return parts
        exemplar = existing[0]
        casted = [_cast_value_like(exemplar, p) for p in parts]
        return casted if isinstance(existing, list) else tuple(casted)

    if existing is None:
        try:
            return _parse_bool(raw)
        except Exception:
            pass
        try:
            return int(raw)
        except Exception:
            pass
        try:
            return float(raw)
        except Exception:
            pass
        if _DATE_RE.match(raw):
            return raw
        return raw

    return raw


def _apply_overrides(settings: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    out = dict(settings)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be KEY=VALUE, got: {ov}")
        k, v = ov.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in out:
            raise KeyError(
                f"Unknown config key: {k}. To keep runs safe, only existing config keys can be overridden."
            )
        out[k] = _cast_value_like(out.get(k), v)
    return out


# ----------------------------
# Paths + compact experiment name
# ----------------------------


def _infer_data_path(settings: Dict[str, Any], cli_data_path: Optional[str]) -> Path:
    if cli_data_path:
        return Path(cli_data_path)
    for key in ("PHASE3_DATA_PATH", "DATA_PATH", "PROCESSED_DATA_PATH"):
        if key in settings and settings[key]:
            return Path(settings[key])
    return Path("data/processed/crsp_daily_with_lagged_funda_phase3.parquet")


def _safe_slug(s: str) -> str:
    s = str(s).replace(" ", "")
    return re.sub(r"[^A-Za-z0-9_=.-]", "_", s)




# --- Compact experiment name helpers ---
def _format_date_token(value: Any) -> str:
    if value is None:
        return "NA"
    return pd.to_datetime(value).strftime("%Y%m%d")


SIGNAL_CODES = {
    "composite_raw_smooth": "crs",
    "composite_raw": "cr",
    "composite_orth": "co",
    "composite_orth_no_idio": "coni",
    "composite_raw_no_idio": "crni",
    "beta_sig": "bs",
    "idiosyncratic_vol": "idv",
    "STR": "str",
    "Residual_mom": "resm",
    "Volatility_AR": "var",
}


DAY_CODES = {
    "MON": "M",
    "TUE": "T",
    "WED": "W",
    "THU": "Th",
    "FRI": "F",
}


COLUMN_CODES = {
    "beta": "beta",
    "sector": "sector",
}


EXCLUDE_FROM_NAME = {
    "SHOW_PROGRESS",
    "PHASE3_DATA_PATH",
    "DATA_PATH",
    "PROCESSED_DATA_PATH",
    "ALLOW_ATTRIBUTION",
    "ALLOW_CAPACITY_ANALYSIS",
    "ALLOW_FACTOR_REGRESSION",
    "ALLOW_MULTI_SIGNAL",
    "ALLOW_OPTIMIZATION",
    "ALLOW_RISK_MODELS",
    "ALLOW_TRANSACTION_COSTS",
    "CRSP_EXCHCD",
    "CRSP_SHRCD",
    "GROSS_LEVERAGE",
    "LIQUIDITY_FILTER_TYPE",
    "LIQUIDITY_RECON_FREQ",
    "LIQUIDITY_TOP_N",
    "NET_EXPOSURE",
    "OPT_COV_LOOKBACK_DAYS",
    "OPT_GROSS_MAX",
    "OPT_LAM",
    "OPT_SHRINK_DELTA",
    "OPT_W_MAX",
    "PORTFOLIO_TYPE",
    "RETURN_HORIZON_DAYS",
    "WEIGHTING_SCHEME",
}


def _format_float_token(value: Any, decimals: int = 4) -> str:
    s = f"{float(value):.{decimals}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _bool_token(value: Any) -> str:
    return "1" if bool(value) else "0"


def _signal_code(signal_col: str) -> str:
    return SIGNAL_CODES.get(signal_col, signal_col)


def _day_code(value: Any) -> str:
    if value is None:
        return "NA"
    return DAY_CODES.get(str(value).upper(), str(value))


def _column_code(value: Any) -> str:
    if value is None:
        return "NA"
    return COLUMN_CODES.get(str(value), str(value))


def _build_run_name(settings: Dict[str, Any], signal_col: str) -> str:
    start = _format_date_token(settings.get("START_DATE"))
    end = _format_date_token(settings.get("END_DATE"))
    freq = settings.get("REBALANCE_FREQUENCY", "NA")
    day = _day_code(settings.get("REBALANCE_DAY"))

    parts = [
        f"sig={_signal_code(signal_col)}",
        f"dt={start}-{end}",
        f"rb={freq}-{day}",
        f"L={_format_float_token(settings.get('LONG_QUANTILE', 0.9), 2)}",
        f"S={_format_float_token(settings.get('SHORT_QUANTILE', 0.1), 2)}",
        f"liq={str(settings.get('LIQUIDITY_FILTER_TYPE', 'NA'))}{int(settings.get('LIQUIDITY_TOP_N', 0))}",
        (
            f"opt={_bool_token(settings.get('USE_OPTIMIZER', False))}"
            f"_bn{_bool_token(settings.get('OPT_BETA_NEUTRAL', False))}{_column_code(settings.get('OPT_BETA_COL'))}"
            f"_sn{_bool_token(settings.get('OPT_SECTOR_NEUTRAL', False))}{_column_code(settings.get('OPT_SECTOR_COL'))}"
            f"_tg{_format_float_token(settings.get('OPT_TURNOVER_GAMMA', 0.0), 4)}"
        ),
        (
            f"sp{_bool_token(settings.get('APPLY_SPREAD_COST', False))}"
            f"_c{_format_float_token(settings.get('SPREAD_CAP', 0.0), 4)}"
        ),
        (
            f"vs{_bool_token(settings.get('APPLY_VOL_SLIPPAGE', False))}"
            f"_l{int(settings.get('VOL_SLIP_LOOKBACK', 0))}"
            f"_k{_format_float_token(settings.get('VOL_SLIP_K', 0.0), 4)}"
            f"_c{_format_float_token(settings.get('VOL_CAP', 0.0), 4)}"
        ),
        (
            f"pc{_bool_token(settings.get('APPLY_PARTICIPATION_CONSTRAINT', False))}"
            f"_a{int(settings.get('ADV_LOOKBACK', 0))}"
            f"_m{_format_float_token(settings.get('MAX_PARTICIPATION', 0.0), 4)}"
        ),
        f"pv={int(float(settings.get('PORTFOLIO_VALUE', 0)))}",
        (
            f"tc{_bool_token(settings.get('APPLY_TURNOVER_COST', False))}"
            f"_{_format_float_token(settings.get('TURNOVER_COST_PER_DOLLAR', 0.0), 6)}"
        ),
    ]

    tracked_base_keys = {
        "START_DATE",
        "END_DATE",
        "REBALANCE_FREQUENCY",
        "REBALANCE_DAY",
        "LONG_QUANTILE",
        "SHORT_QUANTILE",
        "LIQUIDITY_FILTER_TYPE",
        "LIQUIDITY_TOP_N",
        "USE_OPTIMIZER",
        "OPT_BETA_NEUTRAL",
        "OPT_BETA_COL",
        "OPT_SECTOR_NEUTRAL",
        "OPT_SECTOR_COL",
        "OPT_TURNOVER_GAMMA",
        "APPLY_SPREAD_COST",
        "SPREAD_CAP",
        "APPLY_VOL_SLIPPAGE",
        "VOL_SLIP_LOOKBACK",
        "VOL_SLIP_K",
        "VOL_CAP",
        "APPLY_PARTICIPATION_CONSTRAINT",
        "ADV_LOOKBACK",
        "MAX_PARTICIPATION",
        "PORTFOLIO_VALUE",
        "APPLY_TURNOVER_COST",
        "TURNOVER_COST_PER_DOLLAR",
    }

    run_name = _safe_slug("__".join(parts))
    if len(run_name) > 180:
        import hashlib
        digest = hashlib.md5(run_name.encode("utf-8")).hexdigest()[:10]
        run_name = f"{run_name[:165]}__h{digest}"
    return run_name




# ----------------------------
# CLI
# ----------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 baseline backtest runner (CLI override driven)")
    parser.add_argument(
        "--signal",
        required=True,
        help=(
            "Signal name. Either an existing column in the parquet (e.g., be/op/inv) "
            "or the name of a module in src/alphas/ (e.g., STR)."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to parquet (defaults to config path if present, else Phase 3 parquet default).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any existing config key. Repeatable. Example: --set REBALANCE_FREQUENCY=W",
    )
    parser.add_argument(
        "--output-root",
        default="data/outputs/baseline",
        help="Root directory for outputs (run-specific folder will be created under this).",
    )
    parser.add_argument(
        "--use_optimizer",
        default=None,
        help="Optional convenience flag: set USE_OPTIMIZER (0/1/true/false).",
    )

    args = parser.parse_args(argv)

    cfg_mod = _load_project_config_module()
    base_settings = _extract_uppercase_settings(cfg_mod)

    # Ensure baseline keys exist so --set can override them safely
    base_settings.setdefault("REBALANCE_FREQUENCY", "W")
    base_settings.setdefault("REBALANCE_DAY", "FRI")
    base_settings.setdefault("LONG_QUANTILE", 0.9)
    base_settings.setdefault("SHORT_QUANTILE", 0.1)
    base_settings.setdefault("USE_OPTIMIZER", False)
    base_settings.setdefault("SHOW_PROGRESS", False)
    # Phase 9 (component 1): bid-ask spread cost toggles
    base_settings.setdefault("APPLY_SPREAD_COST", False)
    base_settings.setdefault("SPREAD_CAP", 0.20)
    # Phase 9: Setdefaults for new keys so CLI overrides are always allowed
    base_settings.setdefault("APPLY_VOL_SLIPPAGE", False)
    base_settings.setdefault("VOL_SLIP_LOOKBACK", 20)
    base_settings.setdefault("VOL_SLIP_K", 0.10)
    base_settings.setdefault("VOL_CAP", 0.20)

    base_settings.setdefault("APPLY_PARTICIPATION_CONSTRAINT", False)
    base_settings.setdefault("ADV_LOOKBACK", 20)
    base_settings.setdefault("MAX_PARTICIPATION", 0.10)
    base_settings.setdefault("PORTFOLIO_VALUE", 1_000_000.0)

    base_settings.setdefault("APPLY_TURNOVER_COST", False)
    base_settings.setdefault("TURNOVER_COST_PER_DOLLAR", 0.0)

    settings = _apply_overrides(base_settings, overrides=args.set)

    if args.use_optimizer is not None:
        settings["USE_OPTIMIZER"] = _parse_bool(str(args.use_optimizer))

    show_progress = bool(settings.get("SHOW_PROGRESS", False))

    # Runner stage progress (coarse). Uses tqdm if available.
    _pbar = None
    if show_progress:
        if tqdm is not None:
            _pbar = tqdm(total=6, desc="run_baseline", unit="stage")
        else:
            print("[progress] tqdm not installed; falling back to step prints")

    def _advance(stage_msg: str) -> None:
        if not show_progress:
            return
        if _pbar is not None:
            _pbar.set_postfix_str(stage_msg)
            _pbar.update(1)
        else:
            print(f"[progress] {stage_msg}")

    root = _project_root_from_this_file()

    data_path = _infer_data_path(settings, args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = root / out_root

    if not data_path.exists():
        raise FileNotFoundError(f"Parquet not found: {data_path}")

    _advance("load parquet")
    df = pd.read_parquet(data_path)

    # Optional date filtering (speeds up experiments)
    # Controlled via CLI: --set START_DATE=YYYY-MM-DD --set END_DATE=YYYY-MM-DD
    start_date = settings.get("START_DATE")
    end_date = settings.get("END_DATE")

    if start_date is not None:
        sd = pd.to_datetime(start_date)
        df = df[df["date"] >= sd]

    if end_date is not None:
        ed = pd.to_datetime(end_date)
        df = df[df["date"] <= ed]

    if df.empty:
        raise ValueError(
            f"Date filter produced empty df. START_DATE={start_date}, END_DATE={end_date}"
        )

    # ----------------------------
    # Dynamic alpha loading (if not already in parquet)
    # ----------------------------
    _advance("signal ready")
    if args.signal not in df.columns:
        try:
            mod = importlib.import_module(f"src.alphas.{args.signal}")
        except Exception as e:
            raise ImportError(
                f"Signal '{args.signal}' not found as a column in parquet and "
                f"could not import src.alphas.{args.signal}."
            ) from e

        if not hasattr(mod, "compute_signal") and not hasattr(mod, "compute"):
            raise AttributeError(
                f"Alpha module src.alphas.{args.signal} must define compute_signal() or compute()."
            )

        compute_fn = getattr(mod, "compute_signal", None) or getattr(mod, "compute")

        sig_df = compute_fn(df)

        key_cols = {"permno", "date"}
        if not key_cols.issubset(sig_df.columns):
            raise ValueError(
                f"Computed signal must return key columns ['permno','date']; got columns {list(sig_df.columns)}"
            )

        if args.signal not in sig_df.columns:
            value_cols = [c for c in sig_df.columns if c not in key_cols]
            if len(value_cols) == 1:
                sig_df = sig_df.rename(columns={value_cols[0]: args.signal})
            else:
                raise ValueError(
                    f"Computed signal must return columns ['permno','date','{args.signal}']; "
                    f"got columns {list(sig_df.columns)}"
                )

        df = df.merge(sig_df[["permno", "date", args.signal]],
                      on=["permno", "date"], how="left")

    # Guard against duplicate signal columns (for example, beta may already exist
    # in the parquet and also be reintroduced upstream). A duplicate name causes
    # `df[signal]` to return a DataFrame instead of a Series inside the engine.
    signal_matches = [c for c in df.columns if c == args.signal]
    if len(signal_matches) > 1:
        keep_idx = next(i for i, c in enumerate(df.columns) if c == args.signal)
        cols_to_keep = [i for i, c in enumerate(df.columns) if c != args.signal or i == keep_idx]
        df = df.iloc[:, cols_to_keep]

    signal_matches = [c for c in df.columns if c == args.signal]
    if len(signal_matches) != 1:
        raise ValueError(
            f"Expected exactly one '{args.signal}' column before running baseline, found {len(signal_matches)}"
        )

    # ----------------------------
    # Liquidity / universe filter (optional)
    # ----------------------------
    _advance("universe mask")
    # If config.py provides LIQUIDITY_* keys, we build a (date,permno)->in_universe mask
    # and pass it into the engine so ranking/portfolio formation happens inside the universe.
    universe_mask = None
    if settings.get("LIQUIDITY_FILTER_TYPE") and settings.get("LIQUIDITY_TOP_N"):
        spec = UniverseSpec(
            filter_type=str(settings.get("LIQUIDITY_FILTER_TYPE")),
            top_n=int(settings.get("LIQUIDITY_TOP_N")),
            recon_freq=str(settings.get("LIQUIDITY_RECON_FREQ", "M")),
            adv_window=int(settings.get("ADV_WINDOW", 20)),
        )
        try:
            universe_mask = build_liquidity_universe_mask(df, spec)
        except Exception as e:
            raise RuntimeError(
                "Failed to build liquidity universe mask. "
                "Ensure your parquet contains required columns: 'dollar_vol' for ADV or 'mktcap' for MARKET_CAP. "
                f"Underlying error: {e}"
            ) from e

    required = {"permno", "date", "ret_total", args.signal}
    # If liquidity universe filter is enabled, require the relevant liquidity column.
    if settings.get("LIQUIDITY_FILTER_TYPE") and settings.get("LIQUIDITY_TOP_N"):
        if str(settings.get("LIQUIDITY_FILTER_TYPE")) == "ADV":
            required.add("dollar_vol")
        elif str(settings.get("LIQUIDITY_FILTER_TYPE")) == "MARKET_CAP":
            required.add("mktcap")

    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in parquet: {sorted(missing)}")

    _advance("engine")
    # Run engine
    res = run_baseline_backtest(
        df=df,
        settings=settings,
        signal_col=args.signal,
        universe_mask=universe_mask,
        id_col="permno",
        date_col="date",
        ret_col="ret_total",
    )

    run_name = _build_run_name(settings=settings, signal_col=args.signal)
    out_dir = out_root / run_name

    # Overwrite existing run output to avoid accumulating many folders
    if out_dir.exists():
        shutil.rmtree(out_dir)

    _advance("write report")
    # Write report artifacts
    write_report(
        out_dir=out_dir,
        period_returns=res.period_returns,
        turnover=res.turnover,
        equity_curve=res.equity_curve,
        metrics=res.metrics,
        settings_used=settings,
        weights=getattr(res, "weights", None),
        period_details=getattr(res, "period_details", None),
        plot_title="Baseline Equity Curve",
    )

    _advance("done")
    if _pbar is not None:
        _pbar.close()

    print("\n================ BASELINE BACKTEST (Phase 4) ================")
    print(f"Data: {data_path}")
    print(f"Output dir: {out_dir}")
    print(f"Run name: {run_name}")
    print("-------------------------------------------------------------")
    for k in ["periods", "periods_per_year", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"]:
        if k in res.metrics:
            v = res.metrics[k]
            if isinstance(v, (int, float)):
                print(f"{k:>18}: {v:.6f}")
            else:
                print(f"{k:>18}: {v}")
    print("=============================================================\n")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())