from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/train.csv")
PROCESSED_PATH = Path("data/processed/sales_clean.csv")


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns={"store": "store_id", "item": "product_id", "sales": "sales_qty"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df):
    df = df.copy()
    df["store_id"] = "S" + df["store_id"].astype(str).str.zfill(3)
    df["product_id"] = "P" + df["product_id"].astype(str).str.zfill(3)
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"])
    if (df["sales_qty"] < 0).any():
        raise ValueError("Negative sales found")
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return df


def fill_missing_dates(df, method="zero"):
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    pieces = []
    for (store, product), g in df.groupby(["store_id", "product_id"]):
        g = g.set_index("date").reindex(full_range)
        g["store_id"] = store
        g["product_id"] = product
        if method == "ffill":
            g["sales_qty"] = g["sales_qty"].ffill()
        g["sales_qty"] = g["sales_qty"].fillna(0)
        pieces.append(g.rename_axis("date").reset_index())
    return pd.concat(pieces, ignore_index=True)


def time_split(df, train_frac=0.70, val_frac=0.15):
    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


if __name__ == "__main__":
    df = fill_missing_dates(clean(load_raw()))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Rows:", len(df))
    print("Series (store x product):", df.groupby(["store_id", "product_id"]).ngroups)
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Missing values:\n", df.isna().sum())

    train, val, test = time_split(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {part['date'].min().date()} -> {part['date'].max().date()} ({len(part)} rows)")