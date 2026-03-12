import pandas as pd
import numpy as np

data_path = "data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet"
out_path = "data/processed/crsp_daily_with_lagged_funda_phase3_plus_validated_signals.parquet"

df = pd.read_parquet(data_path)

df["date"] = pd.to_datetime(df["date"])

# compute market return (cross-sectional mean)
market = df.groupby("date")["ret_total"].mean()
market.name = "market_ret"

df = df.merge(market, on="date")

window = 252

def compute_beta(group):
    r = group["ret_total"]
    m = group["market_ret"]

    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()

    beta = cov / var
    return beta

df["beta_mkt"] = df.groupby("permno", group_keys=False).apply(compute_beta)

df.to_parquet(out_path, index=False)

print("Saved:", out_path)