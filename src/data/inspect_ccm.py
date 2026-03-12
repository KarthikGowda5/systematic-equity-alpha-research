import pandas as pd

# Load CCM link table (adjust path if needed)
ccm = pd.read_csv("data/raw/CCM_linktable.csv", low_memory=False)

# Normalize column names (WRDS exports often vary in casing / spacing)
ccm.columns = ccm.columns.str.strip().str.lower()

# Common WRDS alias handling
rename_map = {}
if "lpermno" in ccm.columns and "permno" not in ccm.columns:
    rename_map["lpermno"] = "permno"
if "lpermco" in ccm.columns and "permco" not in ccm.columns:
    rename_map["lpermco"] = "permco"
ccm = ccm.rename(columns=rename_map)

required = {"gvkey", "permno", "linktype", "linkdt", "linkenddt"}
missing = sorted(list(required - set(ccm.columns)))
if missing:
    raise KeyError(
        f"Missing required CCM columns: {missing}. "
        f"Available columns: {list(ccm.columns)}"
    )

# ==============================
# CCM LINK TABLE CLEANING
# ==============================

# 1) Keep only LU / LC link types
ccm = ccm[ccm["linktype"].isin(["LU", "LC"])].copy()
# WRDS uses 'E' to denote open-ended link end dates
ccm["linkenddt"] = ccm["linkenddt"].replace("E", pd.NA)
# 2) Convert link dates to datetime
ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])

# 3) Drop rows with missing link start date (structurally invalid)
ccm = ccm[ccm["linkdt"].notna()].copy()

# 4) Replace open-ended links with far-future date
ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

# 5) Check for invalid date ranges (end before start)
invalid_date_ranges = ccm[ccm["linkenddt"] < ccm["linkdt"]]
print("Invalid date ranges:", len(invalid_date_ranges))

# If any invalid rows exist, inspect before dropping
if len(invalid_date_ranges) > 0:
    print(invalid_date_ranges.head())

# 6) Drop exact duplicate rows
ccm = ccm.drop_duplicates().copy()
ccm.to_parquet("data/processed/ccm_lu_lc_clean.parquet", index=False)
# ==============================
# Structural Diagnostics
# ==============================

print("Total rows:", len(ccm))
print("Unique permnos:", ccm["permno"].nunique())
print("Unique gvkeys:", ccm["gvkey"].nunique())
print("Linktype distribution:")
print(ccm["linktype"].value_counts())