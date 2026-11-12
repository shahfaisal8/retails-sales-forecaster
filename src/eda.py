from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_PATH = Path("data/processed/sales_clean.csv")
FIG_DIR = Path("docs/figures")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    daily = df.groupby("date")["sales_qty"].sum()

    # 1. Overall trend
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax, linewidth=0.6, label="Daily total")
    daily.rolling(30).mean().plot(ax=ax, linewidth=2, label="30-day average")
    ax.set(title="Total daily sales (all stores, all products)", ylabel="Units")
    ax.legend()
    save(fig, "01_total_daily_sales.png")

    # 2. Weekday pattern
    by_dow = df.assign(dow=df["date"].dt.day_name()).groupby("dow")["sales_qty"].mean().reindex(DAYS)
    fig, ax = plt.subplots(figsize=(8, 4))
    by_dow.plot.bar(ax=ax)
    ax.set(title="Average sales by day of week", ylabel="Units per store-product")
    save(fig, "02_weekday_pattern.png")

    # 3. Month pattern
    by_month = df.groupby(df["date"].dt.month)["sales_qty"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_month.plot.bar(ax=ax)
    ax.set(title="Average sales by month", xlabel="Month", ylabel="Units per store-product")
    save(fig, "03_month_pattern.png")

    # 4. One example series, daily and 7-day average
    one = df[(df["store_id"] == "S001") & (df["product_id"] == "P001")].set_index("date")["sales_qty"]
    fig, ax = plt.subplots(figsize=(12, 4))
    one.plot(ax=ax, linewidth=0.6, label="Daily")
    one.rolling(7).mean().plot(ax=ax, linewidth=2, label="7-day average")
    ax.set(title="Store S001, Product P001", ylabel="Units")
    ax.legend()
    save(fig, "04_example_series.png")

    # 5. Informal autocorrelation, trend removed
    detrended = (daily - daily.rolling(30, center=True).mean()).dropna()
    acf = [detrended.autocorr(lag=k) for k in range(1, 31)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, 31), acf)
    ax.set(title="Autocorrelation of daily total (trend removed)", xlabel="Lag (days)", ylabel="Correlation")
    save(fig, "05_autocorrelation.png")

    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()