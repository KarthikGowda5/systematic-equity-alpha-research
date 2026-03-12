import pandas as pd
import numpy as np

# ==============================
# LOAD INPUTS (adjust paths)
# ==============================
crsp = pd.read_parquet("data/processed/crsp_phase1.parquet")   # must contain: permno, date
ccm  = pd.read_parquet("data/processed/ccm_lu_lc_clean.parquet")   # from step (2)

# Ensure date types
crsp["date"] = pd.to_datetime(crsp["date"])
ccm["linkdt"] = pd.to_datetime(ccm["linkdt"])
ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"])

# (Optional but recommended) keep only needed CCM cols to reduce memory
keep_cols = ["permno", "gvkey", "linktype", "linkprim", "linkdt", "linkenddt"]
ccm = ccm[[c for c in keep_cols if c in ccm.columns]].copy()

# ==============================
# MEMORY-SAFE DATE-VALID LINKING (no cartesian merge)
# For each permno, pick a valid gvkey where:
#   linkdt <= date <= linkenddt
# Tie-break rule implemented efficiently:
#   - Prefer linkprim in {P, C} when available
#   - Otherwise use any LU/LC link
#   - Within each set, use the most recent linkdt
# ==============================

# Keep CCM only for permnos present in CRSP to reduce work
crsp_permnos = pd.Index(crsp["permno"].unique())
ccm = ccm[ccm["permno"].isin(crsp_permnos)].copy()

# Rank linkprim (lower is better). We'll compute two mappings:
#   1) all LU/LC links
#   2) only primary links (P/C) to override when available
if "linkprim" in ccm.columns:
    prim_rank = {"P": 0, "C": 1}
    ccm["_prim_rank"] = ccm["linkprim"].map(prim_rank).fillna(9).astype(int)
else:
    ccm["_prim_rank"] = 9

# Ensure sorting for group operations
ccm = ccm.sort_values(["permno", "linkdt"])

# Helper: vectorized "latest linkdt <= date" within a permno, then validate linkenddt
def _map_links_one_permno(dates: np.ndarray, linkdt: np.ndarray, linkenddt: np.ndarray, gvkey: np.ndarray):
    # dates, linkdt, linkenddt are datetime64[ns] arrays, sorted by linkdt
    idx = np.searchsorted(linkdt, dates, side="right") - 1
    out = np.full(dates.shape[0], np.nan, dtype=object)
    valid = idx >= 0
    if valid.any():
        idxv = idx[valid]
        # date-validity check
        dv = dates[valid]
        ok = dv <= linkenddt[idxv]
        if ok.any():
            out_idx = np.where(valid)[0][ok]
            out[out_idx] = gvkey[idxv[ok]]
    return out

# Prepare CRSP sorted for deterministic output
tmp = crsp.sort_values(["permno", "date"]).copy()

# Allocate output arrays
gv_all = np.full(len(tmp), np.nan, dtype=object)
gv_pc  = np.full(len(tmp), np.nan, dtype=object)

# Build per-permno index ranges for CRSP to avoid groupby.apply overhead
permno_values = tmp["permno"].to_numpy()
date_values = tmp["date"].to_numpy(dtype="datetime64[ns]")

# Pre-split CCM into "all" and "primary"
ccm_all = ccm.copy()
ccm_pc = ccm[ccm["_prim_rank"].isin([0, 1])].copy()

# Build a dict of CCM links per permno for fast lookup
def _build_link_dict(df: pd.DataFrame):
    d = {}
    for p, g in df.groupby("permno", sort=False):
        d[p] = (
            g["linkdt"].to_numpy(dtype="datetime64[ns]"),
            g["linkenddt"].to_numpy(dtype="datetime64[ns]"),
            g["gvkey"].astype(object).to_numpy(),
        )
    return d

links_all = _build_link_dict(ccm_all)
links_pc  = _build_link_dict(ccm_pc)

# Iterate permno blocks in CRSP (memory-safe)
# Find boundaries where permno changes (tmp is sorted by permno,date)
boundaries = np.flatnonzero(np.r_[True, permno_values[1:] != permno_values[:-1], True])
# boundaries marks starts; last element is len(tmp)
for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i + 1]
    p = permno_values[start]
    dates = date_values[start:end]

    if p in links_all:
        ldt, ledt, gvk = links_all[p]
        gv_all[start:end] = _map_links_one_permno(dates, ldt, ledt, gvk)

    if p in links_pc:
        ldt, ledt, gvk = links_pc[p]
        gv_pc[start:end] = _map_links_one_permno(dates, ldt, ledt, gvk)

# Prefer primary-link mapping where available, else fallback to all-links mapping
tmp["gvkey"] = np.where(pd.isna(gv_pc), gv_all, gv_pc)

# Drop helper col from CCM to keep namespace clean
if "_prim_rank" in ccm.columns:
    ccm = ccm.drop(columns=["_prim_rank"])

# ==============================
# 4) Diagnostics (mandatory)
# ==============================
# Coverage: % of CRSP rows that got a gvkey after date-valid linking
coverage = tmp["gvkey"].notna().mean()

# Uniqueness: ensure one row per (permno, date)
dup_ct = tmp.duplicated(subset=["permno", "date"]).sum()

print("=== CRSP→CCM DATE-VALID MERGE DIAGNOSTICS ===")
print(f"Rows after date-valid linking: {len(tmp):,}")
print(f"Coverage (gvkey non-null): {coverage:.4%}")
print(f"Duplicates on (permno,date): {dup_ct}")

# Optional: how many distinct gvkeys per permno (post-filter, post-tie-break)
# (Should usually be 1 per date, but across time can change)
print("Unique permnos:", tmp["permno"].nunique())
print("Unique gvkeys:", tmp["gvkey"].nunique())

# ==============================
# 5) Save output (ready for fundamentals as-of merge later)
# ==============================
tmp.to_parquet("data/processed/crsp_daily_with_gvkey_phase3.parquet", index=False)
print("Saved: data/processed/crsp_daily_with_gvkey_phase3.parquet")