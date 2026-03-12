import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(path):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_equity(df):
    for col in ["gross", "net_10bps", "net_20bps", "net_model"]:
        df[f"{col}_equity"] = (1 + df[col]).cumprod()
    return df


def plot_equity(df, outdir):
    plt.figure(figsize=(10, 6))

    plt.plot(df["date"], df["gross_equity"], label="Gross")
    plt.plot(df["date"], df["net_10bps_equity"], label="Net 10bps")
    plt.plot(df["date"], df["net_20bps_equity"], label="Net 20bps")
    plt.plot(df["date"], df["net_model_equity"], label="Net Model")

    plt.title("Cost Sensitivity Equity Curves")
    plt.legend()
    plt.tight_layout()

    out = outdir / "equity_curve_sensitivity.png"
    plt.savefig(out)
    plt.close()

    print("Saved:", out)


def plot_cost_vs_turnover(df, outdir):
    plt.figure(figsize=(7, 5))

    plt.scatter(df["turnover"], df["model_cost"], alpha=0.5)

    plt.xlabel("Turnover")
    plt.ylabel("Model Cost")
    plt.title("Cost vs Turnover")

    plt.tight_layout()

    out = outdir / "cost_vs_turnover.png"
    plt.savefig(out)
    plt.close()

    print("Saved:", out)


def save_cost_stats(df, outdir):
    stats = df[["model_cost", "turnover"]].describe()

    out = outdir / "cost_stats.csv"
    stats.to_csv(out)

    print("\nCost Statistics")
    print(stats)

    print("\nSaved:", out)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="net_sims csv file from simulate_nets.py",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="output report directory",
    )

    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = build_equity(df)

    plot_equity(df, outdir)
    plot_cost_vs_turnover(df, outdir)
    save_cost_stats(df, outdir)

    print("\nPhase 9 Sensitivity Report Complete")


if __name__ == "__main__":
    main()
    