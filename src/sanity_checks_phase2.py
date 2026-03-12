import numpy as np
import pandas as pd

# ============================================================
# Phase 2 (Sanity): Fundamentals Table Checks
# Input:  data/processed/funda_bopi.csv
# Checks:
#   - key uniqueness (gvkey, available_date)
#   - date ranges
#   - missingness
#   - inf / NaN issues
#   - distribution diagnostics
# ============================================================

INPUT_PATH = "data/processed/funda.parquet"

core_cols = ["be", "op", "inv", "at", "at_lag", "operating_income"]
key_cols = ["gvkey", "available_date"]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)

    print("\n" + "=" * 60)
    print("BASIC STRUCTURE")
    print("=" * 60)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n" + "=" * 60)
    print("KEY CHECKS")
    print("=" * 60)
    dup_key = df.duplicated(key_cols).sum()
    print("Duplicate (gvkey, available_date):", dup_key)

    print("\n" + "=" * 60)
    print("DATE RANGE")
    print("=" * 60)
    print("Min datadate      :", df["datadate"].min())
    print("Max datadate      :", df["datadate"].max())
    print("Min available_date:", df["available_date"].min())
    print("Max available_date:", df["available_date"].max())

    print("\n" + "=" * 60)
    print("COVERAGE")
    print("=" * 60)
    print("Unique gvkeys:", df["gvkey"].nunique())
    print("Obs per fyear (tail):")
    if "fyear" in df.columns:
        print(df["fyear"].value_counts().sort_index().tail(10))

    print("\n" + "=" * 60)
    print("MISSINGNESS (Overall %)")
    print("=" * 60)
    cols_present = [c for c in core_cols if c in df.columns]
    miss = df[cols_present].isna().mean().sort_values(ascending=False)
    print(miss.round(4))

    print("\n" + "=" * 60)
    print("NUMERIC VALIDITY")
    print("=" * 60)
    for c in ["op", "inv"]:
        if c in df.columns:
            print(f"{c} inf count:", np.isinf(df[c]).sum())

    if "be" in df.columns:
        print("BE <= 0:", round((df["be"] <= 0).mean(), 4))
        print("BE == 0:", int((df["be"] == 0).sum()))

    if "at_lag" in df.columns:
        print("AT_lag <= 0:", round((df["at_lag"] <= 0).mean(), 4))
        print("AT_lag == 0:", int((df["at_lag"] == 0).sum()))

    print("\n" + "=" * 60)
    print("DISTRIBUTIONS")
    print("=" * 60)
    for c in ["be", "op", "inv"]:
        if c in df.columns:
            print(f"\n{c} describe:")
            print(df[c].describe())

    print("\n" + "=" * 60)
    print("QUANTILES (robust tails)")
    print("=" * 60)
    q = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    for c in ["be", "op", "inv", "at", "at_lag"]:
        if c in df.columns:
            print(f"\n{c} quantiles:")
            print(df[c].quantile(q))

    print("\n" + "=" * 60)
    print("EFFECTIVE-DATE LAG SANITY")
    print("=" * 60)
    print(df[["datadate", "available_date"]].head())
    print("\nLag months (integer) head:")
    # Robust month difference: (year, month) arithmetic
    month_diff = (
        (df["available_date"].dt.year - df["datadate"].dt.year) * 12
        + (df["available_date"].dt.month - df["datadate"].dt.month)
    )
    print(month_diff.head())
    print("Unique month diffs:", sorted(month_diff.unique().tolist())[:10])


if __name__ == "__main__":
    main()