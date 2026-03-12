# ============================================================
# Phase 0 Research Contract — DO NOT MODIFY WITHOUT VERSIONING
# ============================================================

from datetime import date

# ------------------------
# Universe Definition
# ------------------------

# CRSP Share Codes: Common shares only
CRSP_SHRCD = [10, 11]

# CRSP Exchange Codes: NYSE, AMEX, NASDAQ
CRSP_EXCHCD = [1, 2, 3]

# Liquidity filter
LIQUIDITY_FILTER_TYPE = "ADV"   # Options: "ADV", "MARKET_CAP"
LIQUIDITY_TOP_N = 1000
LIQUIDITY_RECON_FREQ = "M"      # Monthly reconstitution

# ------------------------
# Time Structure
# ------------------------

START_DATE = date(2000, 1, 1)
END_DATE   = date(2024, 12, 31)

REBALANCE_FREQUENCY = "W"        # Weekly
REBALANCE_DAY = "FRI"            # End-of-week close

RETURN_HORIZON_DAYS = 5          # Next-week return (t → t+5)

# ------------------------
# Portfolio Definition (Baseline)
# ------------------------

PORTFOLIO_TYPE = "MARKET_NEUTRAL"

LONG_QUANTILE = 0.90             # Top 10%
SHORT_QUANTILE = 0.10            # Bottom 10%

WEIGHTING_SCHEME = "EQUAL_DOLLAR"

GROSS_LEVERAGE = 2.0             # 1x long + 1x short
NET_EXPOSURE = 0.0               # Dollar neutral only

# ------------------------
# Explicit Phase 0–2 Restrictions
# ------------------------

ALLOW_OPTIMIZATION = False
ALLOW_MULTI_SIGNAL = False
ALLOW_TRANSACTION_COSTS = False
ALLOW_FACTOR_REGRESSION = False
ALLOW_RISK_MODELS = False
ALLOW_ATTRIBUTION = False
ALLOW_CAPACITY_ANALYSIS = False

# ------------------------
# Phase 8 — Optimization Settings
# ------------------------

OPT_COV_LOOKBACK_DAYS = 252
OPT_SHRINK_DELTA = 0.2
OPT_LAM = 1.0
OPT_W_MAX = 0.01
OPT_GROSS_MAX = 2.0
OPT_TURNOVER_GAMMA = 0.0

OPT_BETA_NEUTRAL = False
OPT_BETA_COL = "beta"

# Sector neutrality (optional)
OPT_SECTOR_NEUTRAL = False
OPT_SECTOR_COL = "sector"

# ------------------------
# Phase 9 — Transaction Cost Model & Execution
# ------------------------

APPLY_SPREAD_COST = False        # Enable bid-ask spread cost model
SPREAD_CAP = 0.20                # Maximum allowed relative spread (20%)

APPLY_VOL_SLIPPAGE = False
VOL_SLIP_LOOKBACK = 20
VOL_SLIP_K = 0.10
VOL_CAP = 0.20

APPLY_PARTICIPATION_CONSTRAINT = False
ADV_LOOKBACK = 20
MAX_PARTICIPATION = 0.10
PORTFOLIO_VALUE = 1_000_000.0

APPLY_TURNOVER_COST = False
TURNOVER_COST_PER_DOLLAR = 0.0

# ------------------------
# Runtime / Diagnostics
# ------------------------

SHOW_PROGRESS = False
