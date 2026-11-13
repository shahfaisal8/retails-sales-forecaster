import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import PROCESSED_PATH, time_split
from features import add_features

NUMERIC = ["lag_1", "lag_7", "lag_14", "lag_28", "ma_7", "ma_14", "is_weekend"]
CATEGORICAL = ["dow", "month", "store_num", "product_num"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "sales_qty"


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def mape(y, p):
    nonzero = y > 0
    return float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100)


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum() * 100)


def report(name, y, p):
    return {"model": name, "MAE": mae(y, p), "MAPE_%": mape(y, p), "WAPE_%": wape(y, p)}


def build_linear():
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("prep", prep), ("model", LinearRegression())])


def build_forest(**overrride):
    params = dict(n_estimators=100,max_depth=16,min_samples_leaf=5,n_jobs=-1,random_state=42)
    params.update(overrride)
    
    return Pipeline([("model", RandomForestRegressor(**params))])


def load_splits():
    df = add_features(pd.read_csv(PROCESSED_PATH, parse_dates=["date"]))
    train, val, test = time_split(df)
    train = train.dropna(subset=FEATURES)
    assert not val[FEATURES].isna().any().any()
    return train, val, test


def main():
    train, val, _ = load_splits()  # the test set stays untouched for now
    y_val = val[TARGET].to_numpy()

    rows = [
        report("naive (lag_1)", y_val, val["lag_1"].to_numpy()),
        report("seasonal naive (lag_7)", y_val, val["lag_7"].to_numpy()),
    ]
    for name, pipe in [("linear regression", build_linear()), ("random forest", build_forest())]:
        pipe.fit(train[FEATURES], train[TARGET])
        rows.append(report(name, y_val, pipe.predict(val[FEATURES])))

    results = pd.DataFrame(rows).round(2)
    print(results.to_string(index=False))
    results.to_csv("docs/results_val.csv", index=False)


if __name__ == "__main__":
    main()