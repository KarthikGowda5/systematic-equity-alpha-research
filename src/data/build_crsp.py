from pathlib import Path
import pandas as pd 
import numpy as np

IN_PATH = Path("data/raw/crsp_dsf_2000_2024_raw.csv")
OUT_PATH = Path("data/processed/crsp_phase1.parquet")

def main():
    # Read
    df = pd.read_csv(IN_PATH, parse_dates=["date"], low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # Coerce numeric (CRSP exports often make ret/dlret objects)
    for c in ["prc", "shrout", "vol", "ret", "dlret", "bidlo", "askhi", "bid", "ask"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["prc_abs"] = df["prc"].abs()
    df["shrout_shares"] = df["shrout"] * 1000.0
    df["mktcap"] = df["prc_abs"] * df["shrout_shares"]
    df["dollar_vol"] = df["prc_abs"] * df["vol"]

    # (1+RET)*(1+DLRET)-1 on delisting date; otherwise RET
    df["ret_total"] = np.where(
        df["dlret"].notna(),
        (1.0 + df["ret"].fillna(0.0)) * (1.0 + df["dlret"]) - 1.0,
        df["ret"]
    )
    dlret_non_null = int(df["dlret"].notna().sum()) if "dlret" in df.columns else None

    keep = [
        "permno", "date", "prc", "vol", "mktcap", "dollar_vol", "ret_total",
        "bidlo", "askhi", "bid", "ask",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["permno", "date"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH}")
    print("Rows:", len(df), "Cols:", len(df.columns))
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("Non-null dlret:", dlret_non_null if dlret_non_null is not None else "N/A")
    
if __name__ == "__main__":
    main()