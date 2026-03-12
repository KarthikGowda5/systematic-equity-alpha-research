

"""Run single-signal alpha validation (Phase 5).

This runner:
1) Loads the processed daily panel parquet
2) Computes (or uses) a signal column
3) Builds and applies optional liquidity universe mask (from config.py + --set overrides)
4) Runs the validation suite (rank IC, t-stat, decay, rolling IC, subperiods, regimes)
5) Writes artifacts to data/outputs/alpha_validation/...

Usage examples
--------------
# Validate an existing column in the parquet (e.g., 'op')
python -m src.runs.run_alpha_validation --signal-col op

# Compute momentum from an alpha file (hyphenated filename supported) and validate it
python -m src.runs.run_alpha_validation --alpha-file src/alphas/momentum12-1.py --alpha-fn compute_momentum_12_1 --signal-col mom_12_1

# Override config values
python -m src.runs.run_alpha_validation --signal-col mom_12_1 --set LIQUIDITY_TOP_N=1000 --set LIQUIDITY_FILTER_TYPE=ADV
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config as project_config
from src.signals.alpha_validation import (
    RegimeSpec,
    ValidationSpec,
    realized_vol_regime_rule,
    run_validation,
)
from src.backtest.universe import UniverseSpec, build_liquidity_universe_mask


def load_settings() -> Dict[str, Any]:
    """Load uppercase variables from config.py into a dict."""
    out: Dict[str, Any] = {}
    for k in dir(project_config):
        if k.isupper():
            out[k] = getattr(project_config, k)
    return out


def coerce_value(v: str) -> Any:
    """Best-effort type coercion for --set KEY=VALUE."""
    s = v.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def apply_overrides(settings: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    out = dict(settings)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Use KEY=VALUE.")
        k, v = item.split("=", 1)
        out[k.strip().upper()] = coerce_value(v)
    return out


def load_alpha_function(alpha_file: str, alpha_fn: str):
    """Load a function from a python file path (supports hyphenated filenames)."""
    path = Path(alpha_file)
    if not path.exists():
        raise FileNotFoundError(f"alpha_file not found: {alpha_file}")

    mod_name = f"alpha_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import alpha module from: {alpha_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    fn = getattr(module, alpha_fn, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Function '{alpha_fn}' not found (or not callable) in {alpha_file}")
    return fn


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5: Single-signal alpha validation")

    parser.add_argument(
        "--data",
        default=getattr(project_config, "PANEL_PATH", "data/processed/crsp_daily_with_lagged_funda_phase3.parquet"),
        help="Path to panel parquet (default: config.PANEL_PATH if present, else Phase 3 panel)",
    )
    parser.add_argument(
        "--signal-col",
        required=True,
        help="Name of the signal column to validate (existing or produced by --alpha-file).",
    )

    # Optional: compute a signal by loading a function from a file.
    parser.add_argument(
        "--alpha-file",
        default=None,
        help="Path to alpha python file that computes the signal (optional).",
    )
    parser.add_argument(
        "--alpha-fn",
        default=None,
        help="Function name inside --alpha-file to call (optional).",
    )

    # Validation knobs
    parser.add_argument("--horizons", default="1-20", help="Horizon range like '1-20' or comma list '1,5,10'")
    parser.add_argument("--rolling", type=int, default=252, help="Rolling window length in trading days")
    parser.add_argument("--nw-lags", type=int, default=5, help="Newey-West lags for IC t-stats")

    # Optional: subperiods via JSON-ish string in one arg to keep runner simple
    parser.add_argument(
        "--subperiods",
        default=None,
        help="Optional JSON list of [start,end,label] triples, e.g. '[[""2000-01-01"",""2009-12-31"",""2000s""]]'",
    )

    # Optional: regime split
    parser.add_argument("--use-vol-regime", action="store_true", help="Enable realized-vol high/low regime split")
    parser.add_argument("--vol-window", type=int, default=20, help="Realized vol window (days) for regime")
    parser.add_argument("--vol-split", type=float, default=0.5, help="Split quantile for high-vol regime (0.5=median)")

    # Config overrides
    parser.add_argument("--set", action="append", default=[], help="Override config KEY=VALUE (repeatable)")

    args = parser.parse_args(argv)

    settings = apply_overrides(load_settings(), args.set)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Panel parquet not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Compute signal if requested
    if args.alpha_file and args.alpha_fn:
        fn = load_alpha_function(args.alpha_file, args.alpha_fn)
        sig_df = fn(df)
        if not isinstance(sig_df, pd.DataFrame):
            raise TypeError("Alpha function must return a pandas DataFrame")
        key_cols = {"permno", "date"}
        if not key_cols.issubset(sig_df.columns):
            raise KeyError(
                f"Alpha output must include key columns ['permno','date']. Got: {list(sig_df.columns)}"
            )

        if args.signal_col not in sig_df.columns:
            value_cols = [c for c in sig_df.columns if c not in key_cols]
            if len(value_cols) == 1:
                sig_df = sig_df.rename(columns={value_cols[0]: args.signal_col})
            else:
                raise KeyError(
                    f"Alpha output must include columns ['permno','date','{args.signal_col}']. "
                    f"Got: {list(sig_df.columns)}"
                )

        df = df.merge(sig_df[["permno", "date", args.signal_col]], on=["permno", "date"], how="left")

    # Compute signal from src/alphas/<signal_col>.py if it is not already in the parquet
    # and the user did not provide --alpha-file/--alpha-fn.
    if (args.signal_col not in df.columns) and not (args.alpha_file and args.alpha_fn):
        try:
            mod = importlib.import_module(f"src.alphas.{args.signal_col}")
        except Exception as e:
            raise KeyError(
                f"Signal column '{args.signal_col}' not found in panel, and could not import "
                f"src.alphas.{args.signal_col}. If your alpha file has a hyphenated name, "
                f"use --alpha-file/--alpha-fn instead."
            ) from e

        if not hasattr(mod, "compute_signal") and not hasattr(mod, "compute"):
            raise AttributeError(
                f"Alpha module src.alphas.{args.signal_col} must define compute_signal() or compute()."
            )

        compute_fn = getattr(mod, "compute_signal", None) or getattr(mod, "compute")
        sig_df = compute_fn(df)

        if not isinstance(sig_df, pd.DataFrame):
            raise TypeError("Alpha compute function must return a pandas DataFrame")

        key_cols = {"permno", "date"}
        if not key_cols.issubset(sig_df.columns):
            raise KeyError(
                f"Alpha output must include key columns ['permno','date']. Got: {list(sig_df.columns)}"
            )

        if args.signal_col not in sig_df.columns:
            value_cols = [c for c in sig_df.columns if c not in key_cols]
            if len(value_cols) == 1:
                sig_df = sig_df.rename(columns={value_cols[0]: args.signal_col})
            else:
                raise KeyError(
                    f"Alpha output must include columns ['permno','date','{args.signal_col}']. "
                    f"Got: {list(sig_df.columns)}"
                )

        df = df.merge(sig_df[["permno", "date", args.signal_col]], on=["permno", "date"], how="left")

    # Required columns for validation
    required = {"permno", "date", "ret_total", args.signal_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in panel: {sorted(missing)}")

    # Optional liquidity universe mask from settings
    universe_mask = None
    if settings.get("LIQUIDITY_FILTER_TYPE") and settings.get("LIQUIDITY_TOP_N"):
        ftype = str(settings.get("LIQUIDITY_FILTER_TYPE"))
        if ftype == "ADV" and "dollar_vol" not in df.columns:
            raise KeyError("Liquidity filter ADV requires 'dollar_vol' column in the panel")
        if ftype == "MARKET_CAP" and "mktcap" not in df.columns:
            raise KeyError("Liquidity filter MARKET_CAP requires 'mktcap' column in the panel")

        spec_u = UniverseSpec(
            filter_type=ftype,
            top_n=int(settings.get("LIQUIDITY_TOP_N")),
            recon_freq=str(settings.get("LIQUIDITY_RECON_FREQ", "M")),
            adv_window=int(settings.get("ADV_WINDOW", 20)),
        )
        universe_mask = build_liquidity_universe_mask(df, spec_u)

    # Parse horizons
    if "-" in args.horizons:
        a, b = args.horizons.split("-", 1)
        horizons = tuple(range(int(a), int(b) + 1))
    else:
        horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    vspec = ValidationSpec(horizons=horizons, rolling_window=args.rolling, nw_lags=args.nw_lags)

    # Subperiods
    subperiods = None
    if args.subperiods:
        parsed = json.loads(args.subperiods)
        subperiods = [(s, e, label) for s, e, label in parsed]

    # Regime
    regime = None
    if args.use_vol_regime:
        def _rule(p: pd.DataFrame, idc: str, dc: str, rc: str) -> pd.Series:
            return realized_vol_regime_rule(p, id_col=idc, date_col=dc, ret_col=rc, market_id=None, window=args.vol_window, split=args.vol_split)

        regime = RegimeSpec(name="realized_vol", rule=_rule)

    out = run_validation(
        panel=df,
        signal_col=args.signal_col,
        spec=vspec,
        universe_mask=universe_mask,
        subperiod_splits=subperiods,
        regime_spec=regime,
    )

    # Output directory naming
    out_root = Path("data/outputs/alpha_validation")
    tag_parts = [f"signal={args.signal_col}", f"H={horizons[0]}-{horizons[-1]}", f"ROLL={args.rolling}", f"NW={args.nw_lags}"]
    if universe_mask is not None:
        tag_parts.append(f"U={settings.get('LIQUIDITY_FILTER_TYPE')}_{settings.get('LIQUIDITY_TOP_N')}")
    if regime is not None:
        tag_parts.append("REG=realized_vol")
    out_dir = out_root / "__".join(tag_parts)
    ensure_dir(out_dir)

    # Write artifacts
    out["ic_series"].to_csv(out_dir / "ic_series.csv", header=True)
    out["decay"].to_csv(out_dir / "decay.csv", index=False)
    out["rolling_ic"].to_csv(out_dir / "rolling_ic.csv", header=True)

    with open(out_dir / "ic_summary.json", "w") as f:
        json.dump(out["ic_summary"], f, indent=2, default=str)

    if out.get("subperiods") is not None:
        out["subperiods"].to_csv(out_dir / "subperiods.csv", index=False)

    if out.get("regimes") is not None:
        with open(out_dir / "regimes.json", "w") as f:
            json.dump(out["regimes"], f, indent=2, default=str)

    # Always save settings used
    with open(out_dir / "settings_used.json", "w") as f:
        json.dump(
            {
                "data": str(data_path),
                "signal_col": args.signal_col,
                "alpha_file": args.alpha_file,
                "alpha_fn": args.alpha_fn,
                "validation": {"horizons": list(horizons), "rolling": args.rolling, "nw_lags": args.nw_lags},
                "liquidity": {
                    "enabled": universe_mask is not None,
                    "filter_type": settings.get("LIQUIDITY_FILTER_TYPE"),
                    "top_n": settings.get("LIQUIDITY_TOP_N"),
                    "recon_freq": settings.get("LIQUIDITY_RECON_FREQ"),
                    "adv_window": settings.get("ADV_WINDOW", 20),
                },
                "regime": {"enabled": regime is not None, "name": getattr(regime, "name", None), "vol_window": args.vol_window, "vol_split": args.vol_split},
                "subperiods": subperiods,
            },
            f,
            indent=2,
            default=str,
        )

    # ----------------------------
    # Plots (saved as PNG)
    # ----------------------------
    ic_series = out["ic_series"].dropna().copy()
    if not ic_series.empty:
        # 1) IC series
        plt.figure()
        ic_series.plot()
        plt.title(f"Rank IC (h=1): {args.signal_col}")
        plt.xlabel("Date")
        plt.ylabel("Rank IC")
        plt.tight_layout()
        plt.savefig(out_dir / "ic_series.png", dpi=150)
        plt.close()

        # 2) Rolling IC
        rolling_ic = out["rolling_ic"].dropna().copy()
        if not rolling_ic.empty:
            plt.figure()
            rolling_ic.plot()
            plt.title(f"Rolling Mean Rank IC ({args.rolling}d): {args.signal_col}")
            plt.xlabel("Date")
            plt.ylabel("Rolling Mean Rank IC")
            plt.tight_layout()
            plt.savefig(out_dir / "rolling_ic.png", dpi=150)
            plt.close()

        # 3) Decay curve
        decay = out["decay"].copy()
        if not decay.empty and {"horizon", "mean_rank_ic"}.issubset(decay.columns):
            plt.figure()
            plt.plot(decay["horizon"], decay["mean_rank_ic"], marker="o")
            plt.title(f"Decay Curve (Mean Rank IC): {args.signal_col}")
            plt.xlabel("Horizon (days)")
            plt.ylabel("Mean Rank IC")
            plt.tight_layout()
            plt.savefig(out_dir / "decay.png", dpi=150)
            plt.close()

        # 4) IC histogram
        plt.figure()
        plt.hist(ic_series.values, bins=50)
        plt.title(f"Rank IC Distribution (h=1): {args.signal_col}")
        plt.xlabel("Rank IC")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "ic_hist.png", dpi=150)
        plt.close()

    # ----------------------------
    # Signal page (Markdown)
    # ----------------------------
    mean_ic = float(out["ic_summary"]["mean_rank_ic"])
    nw_t = float(out["ic_summary"]["nw_t"]) if out["ic_summary"]["nw_t"] is not None else float("nan")
    n_obs = int(out["ic_summary"]["n_obs"])

    # Simple pass/fail rules (Phase 5 gate)
    # 1) IC non-trivial and statistically meaningful
    ic_non_trivial = abs(mean_ic) >= 0.002
    ic_significant = abs(nw_t) >= 2.0
    pass_ic = ic_non_trivial and ic_significant

    # 2) Decay sensible: doesn't vanish immediately
    decay_df = out["decay"].copy()
    h5 = decay_df.loc[decay_df["horizon"] == 5, "mean_rank_ic"]
    hmax = decay_df.loc[decay_df["horizon"] == max(horizons), "mean_rank_ic"]
    h5_val = float(h5.iloc[0]) if len(h5) else float("nan")
    hmax_val = float(hmax.iloc[0]) if len(hmax) else float("nan")
    pass_decay = (np.isfinite(h5_val) and h5_val > 0) and (np.isfinite(hmax_val) and hmax_val > 0)

    verdict = "PASS" if (pass_ic and pass_decay) else "FAIL"

    # Definition block (best-effort from runner inputs)
    definition_lines = []
    definition_lines.append(f"**Signal column:** `{args.signal_col}`")
    if args.alpha_file and args.alpha_fn:
        definition_lines.append(f"**Alpha source:** `{args.alpha_file}` (function `{args.alpha_fn}`)")
    definition_lines.append(f"**Data panel:** `{data_path}`")

    if universe_mask is not None:
        definition_lines.append(
            "**Universe:** Liquidity-filtered "
            f"({settings.get('LIQUIDITY_FILTER_TYPE')}, top {settings.get('LIQUIDITY_TOP_N')}, recon {settings.get('LIQUIDITY_RECON_FREQ')}, ADV_WINDOW={settings.get('ADV_WINDOW', 20)})"
        )
    else:
        definition_lines.append("**Universe:** Full panel (no liquidity filter applied)")

    md = []
    md.append(f"# Signal: {args.signal_col}\n")

    md.append("## Definition\n")
    md.extend([f"- {line}" for line in definition_lines])

    md.append("\n## Validation Summary\n")
    md.append(f"- **Mean rank IC (h=1):** {mean_ic:.6f}")
    md.append(f"- **Newey–West t-stat (lags={args.nw_lags}):** {nw_t:.3f}")
    md.append(f"- **IC observations (n):** {n_obs}")
    md.append(f"- **Horizons:** {horizons[0]}–{horizons[-1]} days")
    md.append(f"- **Rolling window:** {args.rolling} trading days\n")

    md.append("## Plots\n")
    md.append("- Rank IC time series (h=1): `ic_series.png`")
    md.append("- Rolling mean Rank IC: `rolling_ic.png`")
    md.append("- Decay curve: `decay.png`")
    md.append("- IC histogram: `ic_hist.png`\n")

    md.append("## Gate Check (Phase 5)\n")
    md.append(f"- IC non-trivial (|mean IC| ≥ 0.002): {'✅' if ic_non_trivial else '❌'}")
    md.append(f"- IC significant (|t| ≥ 2): {'✅' if ic_significant else '❌'}")
    md.append(f"- Decay sensible (mean IC positive at h=5 and h={max(horizons)}): {'✅' if pass_decay else '❌'}")
    md.append(f"\n### Verdict: **{verdict}**\n")

    if verdict == "FAIL":
        md.append("## Failure Mode Notes\n")
        if not pass_ic:
            md.append("- Mean IC and/or t-stat did not meet the non-trivial + significance thresholds.")
        if not pass_decay:
            md.append("- Decay did not remain positive at medium/long horizons (signal may be short-lived or noisy).")

    (out_dir / "signal_page.md").write_text("\n".join(md))

    # Console summary
    print("\n================ ALPHA VALIDATION (Phase 5) ================")
    print(f"Data: {data_path}")
    print(f"Signal: {args.signal_col}")
    print(f"Output dir: {out_dir}")
    print("------------------------------------------------------------")
    s = out["ic_summary"]
    print(f"mean_rank_ic: {s['mean_rank_ic']:.6f}")
    print(f"nw_t       : {s['nw_t']:.3f}   (lags={s['nw_lags']}, n={s['n_obs']})")
    print("============================================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())