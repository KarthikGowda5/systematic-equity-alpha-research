from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("data/processed/crsp_phase1.parquet")

def main():
    df = pd.read_parquet(PATH)

    print("\n================ DUPLICATE CHECK ================")
    dup = df.duplicated(subset=["permno", "date"]).sum()
    print("Duplicate (permno,date):", dup)

    print("\n================ RETURN DISTRIBUTION ================")
    ret = df["ret_total"].dropna()

    print("Mean:", ret.mean())
    print("Std:", ret.std())
    print("Min:", ret.min())
    print("1%:", ret.quantile(0.01))
    print("99%:", ret.quantile(0.99))
    print("Max:", ret.max())

    # Extreme return count
    extreme = (ret.abs() > 1).sum()
    print("Abs(Return) > 100% count:", extreme)
    print("Fraction extreme:", extreme / len(ret))

    marketcap =df.groupby(df["date"].dt.year)["mktcap"].median().iloc[[0,-1]]
    print(marketcap)
    print("\n================ UNIVERSE SIZE ================")
    universe = df.groupby("date")["permno"].nunique()

    print("Min names:", universe.min())
    print("Max names:", universe.max())
    print("Mean names:", universe.mean())

    print("\nLowest universe dates:")
    print(universe.nsmallest(10))

    print("\nSample years:")
    yearly = universe.resample("Y").mean()
    print(yearly.head())
    print(yearly.tail())

    print("\n================ DONE ================")

if __name__ == "__main__":
    main()