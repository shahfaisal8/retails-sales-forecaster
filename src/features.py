import pandas as pd

KEYS = ["store_id", "product_id"]
LAGS = [1, 7, 14, 28]
WINDOWS = [7, 14]


def add_features(df):
    df = df.sort_values(KEYS + ["date"]).reset_index(drop=True)
    sales = df.groupby(KEYS)["sales_qty"]

    for k in LAGS:
        df[f"lag_{k}"] = sales.shift(k)

    for w in WINDOWS:
        df[f"ma_{w}"] = sales.transform(lambda s: s.shift(1).rolling(w).mean())

    df["dow"] = df["date"].dt.day_of_week
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month

    df["store_num"] = df["store_id"].str[1:].astype(int)
    df["product_num"] = df["product_id"].str[1:].astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])
    df = add_features(df)

    one = df[(df["store_id"] == "S001") & (df["product_id"] == "P001")].reset_index(drop=True)
    assert one.loc[10, "lag_1"] == one.loc[9, "sales_qty"]
    assert one.loc[10, "lag_7"] == one.loc[3, "sales_qty"]
    assert abs(one.loc[10, "ma_7"] - one.loc[3:9, "sales_qty"].mean()) < 1e-9
    print("Feature checks passed")
    print(one.head(32).tail(4).T)