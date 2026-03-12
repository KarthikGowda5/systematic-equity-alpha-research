import pandas as pd
import numpy as np

# ==============================
# LOAD INPUTS (adjust paths)
# ==============================
crsp = pd.read_parquet("data/processed/crsp_daily_with_gvkey_phase3.parquet")
funda = pd.read_parquet("data/processed/funda.parquet")

# ==============================
# TYPE / CLEAN
# ==============================
crsp["date"] = pd.to_datetime(crsp["date"])
# Normalize gvkey robustly:
# - Handles numeric exports like 1004.0 (becomes 001004, not 010040)
# - Preserves leading zeros for true gvkeys
crsp_gv_num = pd.to_numeric(crsp["gvkey"], errors="coerce")
crsp["gvkey"] = (
    crsp_gv_num.round(0).astype("Int64").astype(str).replace("<NA>", pd.NA).str.zfill(6)
)

funda["available_date"] = pd.to_datetime(funda["available_date"])
funda["datadate"] = pd.to_datetime(funda["datadate"])
# Normalize gvkey robustly (same logic as CRSP)
funda_gv_num = pd.to_numeric(funda["gvkey"], errors="coerce")
funda["gvkey"] = (
    funda_gv_num.round(0).astype("Int64").astype(str).replace("<NA>", pd.NA).str.zfill(6)
)

# Keep only what you need (reduces memory)
funda = funda[["gvkey", "datadate", "available_date", "be", "op", "inv", "sich", "sector"]].copy()

# ==============================
# KEY OVERLAP DIAGNOSTICS
# ==============================
crsp_g = set(crsp["gvkey"].dropna().unique())
funda_g = set(funda["gvkey"].dropna().unique())
inter = crsp_g.intersection(funda_g)
print(f"Unique CRSP gvkeys: {len(crsp_g):,}")
print(f"Unique FUNDA gvkeys: {len(funda_g):,}")
print(f"Intersection gvkeys : {len(inter):,}")
if len(inter) == 0:
    print("Sample CRSP gvkeys:", list(sorted(crsp_g))[:5])
    print("Sample FUNDA gvkeys:", list(sorted(funda_g))[:5])

crsp_nonnull = crsp[crsp["gvkey"].notna() & (crsp["gvkey"] != "nan")].copy()
crsp_null = crsp[~(crsp["gvkey"].notna() & (crsp["gvkey"] != "nan"))].copy()

# Sort for per-gvkey processing
crsp_nonnull = crsp_nonnull.sort_values(["gvkey", "date"]).copy()
funda = funda.sort_values(["gvkey", "available_date"]).copy()

# ==============================
# VECTORIZED PER-GVKEY AS-OF JOIN (available_date <= date)
# ==============================
# Build dict: gvkey -> arrays of fundamentals, sorted by available_date
fund_map = {}
for g, df in funda.groupby("gvkey", sort=False):
    fund_map[g] = (
        df["available_date"].to_numpy(dtype="datetime64[ns]"),
        df["datadate"].to_numpy(dtype="datetime64[ns]"),
        df["be"].to_numpy(),
        df["op"].to_numpy(),
        df["inv"].to_numpy(),
        df["sich"].to_numpy(),
        df["sector"].to_numpy(),
    )

gv = crsp_nonnull["gvkey"].to_numpy()
dt = crsp_nonnull["date"].to_numpy(dtype="datetime64[ns]")

n = len(crsp_nonnull)
out_avail = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
out_datad = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
out_be = np.full(n, np.nan)
out_op = np.full(n, np.nan)
out_inv = np.full(n, np.nan)
out_sich = np.full(n, np.nan)
out_sector = np.full(n, np.nan)

# boundaries for gvkey blocks (crsp_nonnull is sorted by gvkey,date)
boundaries = np.flatnonzero(np.r_[True, gv[1:] != gv[:-1], True])

for i in range(len(boundaries) - 1):
    start, end = boundaries[i], boundaries[i + 1]
    g = gv[start]
    if g not in fund_map:
        continue

    avail, datad, be_arr, op_arr, inv_arr, sich_arr, sector_arr = fund_map[g]
    dates = dt[start:end]

    idx = np.searchsorted(avail, dates, side="right") - 1
    valid = idx >= 0
    if not valid.any():
        continue

    pos = np.where(valid)[0]
    pick = idx[valid]

    out_avail[start + pos] = avail[pick]
    out_datad[start + pos] = datad[pick]
    out_be[start + pos] = be_arr[pick]
    out_op[start + pos] = op_arr[pick]
    out_inv[start + pos] = inv_arr[pick]
    out_sich[start + pos] = sich_arr[pick]
    out_sector[start + pos] = sector_arr[pick]

crsp_nonnull["available_date"] = out_avail
crsp_nonnull["datadate"] = out_datad
crsp_nonnull["be"] = out_be
crsp_nonnull["op"] = out_op
crsp_nonnull["inv"] = out_inv
crsp_nonnull["sich"] = out_sich
crsp_nonnull["sector"] = out_sector

# Add back null-gvkey rows (no fundamentals)
for col in ["datadate", "available_date", "be", "op", "inv", "sich", "sector"]:
    crsp_null[col] = pd.NA

# Concatenate back rows without gvkey (may be empty); avoid pandas FutureWarning
if len(crsp_null) == 0:
    merged = crsp_nonnull
else:
    crsp_null = crsp_null.reindex(columns=crsp_nonnull.columns)

    # Match dtypes to avoid pandas FutureWarning about all-NA entries during concat
    for c in crsp_nonnull.columns:
        try:
            crsp_null[c] = crsp_null[c].astype(crsp_nonnull[c].dtype)
        except Exception:
            # If casting fails (e.g., mixed/object), keep as-is
            pass

    merged = pd.concat([crsp_nonnull, crsp_null], ignore_index=True)

# ==============================
# SANITY CHECKS (MANDATORY)
# ==============================
bad = merged["available_date"].notna() & (merged["available_date"] > merged["date"])
print("Look-ahead violations:", int(bad.sum()))

coverage = merged["available_date"].notna().mean()
print(f"Fundamentals coverage (available_date non-null): {coverage:.4%}")

if merged["available_date"].notna().any():
    staleness_days = (merged["date"] - merged["available_date"]).dt.days
    print("Staleness (days) — median:", float(staleness_days.median()))
    print("Staleness (days) — 95%  :", float(staleness_days.quantile(0.95)))

# ==============================
# SAVE
# ==============================
out_path = "data/processed/crsp_daily_with_lagged_funda_phase3.parquet"
merged.to_parquet(out_path, index=False)
print(f"Saved: {out_path}")