from pathlib import Path
import pandas as pd 

CSV_PATH = Path("data/raw/crsp_dsf_2000_2024_raw.csv")
def main():
    # Read (parse date; keep CRSP ret fields as strings first)
    df = pd.read_csv(CSV_PATH, parse_dates=["date"], low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df["dlret"] = pd.to_numeric(df["dlret"], errors="coerce")
    print("\n--- Head ---")
    print(df.head())
    print (len(df))

    print("\n--- Columns ---")
    print(df.columns.tolist())

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Date range ---")
    print(df["date"].min(), "->", df["date"].max())

    # Key uniqueness check
    dup = df.duplicated(subset=["permno", "date"]).sum()
    print("\n--- Duplicates (permno,date) ---")
    print(dup)

    print("\n--- Missingness (top 20) ---")
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    print(miss)

if __name__ == "__main__":
    main()