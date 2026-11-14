import numpy as np
import pandas as pd

from src.train import FEATURES

SHORT_HISTORY_DAYS = 30
SHORT_HISTORY_MESSAGE = "Not enough history; forecasts may be unreliable"


def _lag(vals, k):
    # If the history is shorter than k days, use the average as a stand-in
    return vals[-k] if len(vals) >= k else float(np.mean(vals))


def feature_row(vals, day, store_num, product_num):
    """Features for predicting `day`, given past sales `vals` (oldest first)."""
    dow = day.dayofweek
    return {
        "lag_1": _lag(vals, 1),
        "lag_7": _lag(vals, 7),
        "lag_14": _lag(vals, 14),
        "lag_28": _lag(vals, 28),
        "ma_7": float(np.mean(vals[-7:])),
        "ma_14": float(np.mean(vals[-14:])),
        "is_weekend": int(dow >= 5),
        "dow": dow,
        "month": day.month,
        "store_num": store_num,
        "product_num": product_num,
    }


def forecast_series(pipe, history, store_num, product_num, horizon=30, band=(10, 90)):
    """Recursive forecast for ONE series. `history` needs columns date, sales_qty
    with no missing days (run fill_missing_dates first). Returns (forecast_df, warning)."""
    if history is None or len(history) == 0:
        raise ValueError("History is empty: cannot forecast")
    history = history.sort_values("date")
    vals = history["sales_qty"].astype(float).tolist()
    warning = SHORT_HISTORY_MESSAGE if len(vals) < SHORT_HISTORY_DAYS else None

    forest = pipe.named_steps["model"]
    day = history["date"].max()
    rows = []
    for _ in range(horizon):
        day = day + pd.Timedelta(days=1)
        x = pd.DataFrame([feature_row(vals, day, store_num, product_num)])[FEATURES].to_numpy()
        tree_preds = np.array([tree.predict(x)[0] for tree in forest.estimators_])
        point = max(float(tree_preds.mean()), 0.0)
        rows.append({
            "date": day,
            "forecast": point,
            "lower": max(float(np.percentile(tree_preds, band[0])), 0.0),
            "upper": float(np.percentile(tree_preds, band[1])),
        })
        vals.append(point)  # feed our own prediction back in as tomorrow's "history"
    return pd.DataFrame(rows), warning


def forecast_next_day(pipe, df):
    """Next-day forecast for EVERY store-product series (used by the daily batch job)."""
    rows, meta = [], []
    for (store, product), g in df.sort_values("date").groupby(["store_id", "product_id"]):
        vals = g["sales_qty"].astype(float).tolist()
        day = g["date"].max() + pd.Timedelta(days=1)
        rows.append(feature_row(vals, day, int(store[1:]), int(product[1:])))
        meta.append({"date": day, "store_id": store, "product_id": product})
    out = pd.DataFrame(meta)
    out["forecast"] = np.clip(pipe.predict(pd.DataFrame(rows)[FEATURES]), 0, None)
    return out