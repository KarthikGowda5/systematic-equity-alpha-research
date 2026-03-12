import os

import numpy as np
import pandas as pd

# ============================================================
# Phase 2 (Build): Compustat Annual Fundamentals
# Outputs a fundamentals table keyed by (gvkey, available_date)
# with a strict 6-month reporting lag.
#
# Computes:
#   - be     : Book Equity
#   - op     : Operating Profitability
#   - inv    : Investment (asset growth)
#   - sich   : Compustat historical SIC (firm-level)
#   - sector : Coarse sector bucket derived from SIC (floor(sich/1000))
#
# NOTE: This file performs ONLY construction + save.
#       Diagnostics belong in src/sanity_checks_funda_phase2.py
# ============================================================

INPUT_PATH = "data/raw/funda.csv"
OUTPUT_PATH = "data/processed/funda.parquet"

# Mechanical denominator floors to avoid ratio explosions
BE_FLOOR = 10.0
AT_LAG_FLOOR = 10.0


def build_funda_phase2(input_path: str = INPUT_PATH) -> pd.DataFrame:
    """Load Compustat funda.csv and compute BE/OP/INV + 6M availability date."""

    funda = pd.read_csv(input_path, parse_dates=["datadate"])

    # Ensure gvkey is stable for downstream merges
    funda["gvkey"] = funda["gvkey"].astype(str).str.zfill(6)

    # Industry classification (firm-level)
    # Compustat historical SIC: `sich` (may be missing in the main funda extract).
    # If missing/all-null, merge from data/raw/sector.csv on (gvkey, datadate).
    if ("sich" not in funda.columns) or funda["sich"].isna().all():
        try:
            sich_path = "data/raw/sector.csv"
            sich_df = pd.read_csv(
                sich_path,
                usecols=["gvkey", "datadate", "sich"],
                parse_dates=["datadate"],
            )
            sich_df["gvkey"] = sich_df["gvkey"].astype(str).str.zfill(6)
            sich_df["sich"] = pd.to_numeric(sich_df["sich"], errors="coerce")
            funda = funda.merge(sich_df, on=["gvkey", "datadate"], how="left")
            print(f"[Phase2] Merged sich from {sich_path}")
        except Exception as e:
            raise RuntimeError(
                f"[Phase2] 'sich' missing/all-null in {INPUT_PATH} and failed to load data/raw/sector.csv: {e}"
            )

    funda["sich"] = pd.to_numeric(funda.get("sich"), errors="coerce")

    # Coarse sector bucket: 0-9 using SIC thousands digit (e.g., 5080 -> 5)
    funda["sector"] = (funda["sich"] // 1000).astype("Int64")

    # Sort for time-series operations
    funda = funda.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

    # ----------------------------
    # Book Equity
    # BE = SEQ + TXDB - PS
    # PS hierarchy: pstkrv -> pstkl -> pstk -> 0
    # ----------------------------
    ps = funda["pstkrv"].fillna(funda["pstkl"]).fillna(funda["pstk"]).fillna(0)
    funda["be"] = funda["seq"] + funda["txdb"].fillna(0) - ps

    # ----------------------------
    # Profitability
    # OP = (REVT - COGS - XSGA - XINT) / BE
    # ----------------------------
    funda["operating_income"] = (
        funda["revt"] - funda["cogs"] - funda["xsga"] - funda["xint"]
    )
    funda["op"] = funda["operating_income"] / funda["be"]

    # ----------------------------
    # Investment (Asset Growth)
    # INV = (AT_t - AT_{t-1}) / AT_{t-1}
    # ----------------------------
    funda["at_lag"] = funda.groupby("gvkey")["at"].shift(1)
    funda["inv"] = (funda["at"] - funda["at_lag"]) / funda["at_lag"]

    # ----------------------------
    # Mechanical cleaning (no look-ahead; just numeric validity)
    # ----------------------------
    # Require positive denominators
    funda.loc[funda["be"] <= 0, "op"] = np.nan
    funda.loc[funda["at_lag"] <= 0, "inv"] = np.nan

    # Replace inf/-inf with NaN
    funda["op"] = funda["op"].replace([np.inf, -np.inf], np.nan)
    funda["inv"] = funda["inv"].replace([np.inf, -np.inf], np.nan)

    # Denominator floors (avoid ratio explosions from tiny denominators)
    funda.loc[funda["be"] < BE_FLOOR, "op"] = np.nan
    funda.loc[funda["at_lag"] < AT_LAG_FLOOR, "inv"] = np.nan

    # ----------------------------
    # 6-Month Reporting Lag
    # available_date = datadate + 6 months
    # ----------------------------
    funda["available_date"] = funda["datadate"] + pd.DateOffset(months=6)

    cols_to_keep = [
        "gvkey",
        "datadate",
        "available_date",
        "be",
        "op",
        "inv",
        "sich",
        "sector",
    ]

    funda = funda[cols_to_keep]
    return funda


def main() -> None:
    funda = build_funda_phase2(INPUT_PATH)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    funda.to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()