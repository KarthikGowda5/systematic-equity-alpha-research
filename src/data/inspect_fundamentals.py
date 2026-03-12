import pandas as pd

# ----------------------------
# Load Data
# ----------------------------
funda = pd.read_csv("data/raw/funda.csv", parse_dates=["datadate"])
funda["gvkey"] = funda["gvkey"].astype(str).str.zfill(6)

print("\n" + "="*60)
print("BASIC STRUCTURE")
print("="*60)
print("Shape:", funda.shape)
print("\nColumns:")
print(funda.columns.tolist())
print("\nData Types:")
print(funda.dtypes)

# ----------------------------
# Duplicate Check
# ----------------------------
print("\n" + "="*60)
print("DUPLICATE CHECK")
print("="*60)
dup_count = funda.duplicated(["gvkey", "datadate"]).sum()
print("Duplicate (gvkey, datadate):", dup_count)

# ----------------------------
# Date Range & Firm Coverage
# ----------------------------
print("\n" + "="*60)
print("DATE RANGE & COVERAGE")
print("="*60)
print("Min datadate:", funda["datadate"].min())
print("Max datadate:", funda["datadate"].max())
print("Unique gvkeys:", funda["gvkey"].nunique())

print("\nObservations per fiscal year:")
print(funda["fyear"].value_counts().sort_index())

# ----------------------------
# Missingness Overview
# ----------------------------
print("\n" + "="*60)
print("MISSINGNESS (Overall %)")
print("="*60)
print(funda.isna().mean().sort_values(ascending=False).round(4))

core_cols = ["at", "seq", "revt", "cogs", "xsga", "xint", "txdb", "pstkrv", "pstkl", "pstk"]
print("\nCore Variable Missingness:")
print(funda[core_cols].isna().mean().round(4))

# ----------------------------
# Missingness by Year
# ----------------------------
print("\n" + "="*60)
print("MISSINGNESS BY FISCAL YEAR")
print("="*60)
miss_by_year = funda.groupby("fyear")[core_cols].apply(lambda x: x.isna().mean())
print(miss_by_year.loc[1995:2024].round(3))

# ----------------------------
# Missingness by Costat
# ----------------------------
print("\n" + "="*60)
print("MISSINGNESS BY COSTAT")
print("="*60)
miss_by_costat = funda.groupby("costat")[core_cols].apply(lambda x: x.isna().mean())
print(miss_by_costat.round(3))

print("\nCostat distribution:")
print(funda["costat"].value_counts(normalize=True).round(4))

# ----------------------------
# Accounting Sanity Checks
# ----------------------------
print("\n" + "="*60)
print("ACCOUNTING SANITY CHECKS")
print("="*60)
print("AT <= 0:", (funda["at"] <= 0).mean())
print("SEQ <= 0:", (funda["seq"] <= 0).mean())

print("\nAT Summary Statistics:")
print(funda["at"].describe())

# Preferred stock availability
ps_any = funda[["pstkrv", "pstkl", "pstk"]].notna().any(axis=1).mean()
print("\nHas any preferred stock field available:", ps_any)