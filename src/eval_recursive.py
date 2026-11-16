import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import PROCESSED_PATH
from src.finalize import DEPLOY_PARAMS
from src.forecast import forecast_series
from src.train import FEATURES, TARGET, build_forest, load_splits

HORIZON = 30
N_SERIES = 40
ORIGINS = ["2016-08-01", "2016-11-01", "2017-02-01"]  # all inside the validation period
BUCKETS = [(1, 7), (8, 14), (15, 21), (22, 30)]


def main():
    train, _, _ = load_splits()  # the test set is not used here
    model = build_forest(**DEPLOY_PARAMS).fit(train[FEATURES], train[TARGET])

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    keys = list(df[["store_id", "product_id"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(0)
    chosen = [keys[i] for i in rng.choice(len(keys), N_SERIES, replace=False)]

    rows = []
    for store, product in chosen:
        series = df[(df["store_id"] == store) & (df["product_id"] == product)].sort_values("date")
        for origin in map(pd.Timestamp, ORIGINS):
            history = series[series["date"] < origin]
            actual = series[(series["date"] >= origin) & (series["date"] < origin + pd.Timedelta(days=HORIZON))]
            if len(actual) < HORIZON:
                continue
            fc, _ = forecast_series(model, history, int(store[1:]), int(product[1:]), HORIZON)
            last_week = history["sales_qty"].to_numpy()[-7:]
            for h in range(1, HORIZON + 1):
                rows.append({
                    "h": h,
                    "origin": str(origin.date()),
                    "actual": actual["sales_qty"].iloc[h - 1],
                    "forecast": fc["forecast"].iloc[h - 1],
                    "naive": last_week[(h - 1) % 7],
                })
        print("done", store, product)

    res = pd.DataFrame(rows)
    res["err_model"] = (res["forecast"] - res["actual"]).abs()
    res["err_naive"] = (res["naive"] - res["actual"]).abs()

    print("\nMAE by forecast horizon (days ahead)")
    print(res.groupby("h")[["err_model", "err_naive"]].mean().loc[[1, 7, 14, 21, 30]].round(2))
    print("\nOverall 30-day MAE: model", round(res["err_model"].mean(), 2),
          "| seasonal naive", round(res["err_naive"].mean(), 2))
    print("Overall 30-day WAPE %: model", round(res["err_model"].sum() / res["actual"].sum() * 100, 2),
          "| seasonal naive", round(res["err_naive"].sum() / res["actual"].sum() * 100, 2))

    

        # Two candidate bands, built from real errors
    res["rel"] = (res["actual"] - res["forecast"]) / res["forecast"].clip(lower=1)
    res["z"] = (res["actual"] - res["forecast"]) / np.sqrt(res["forecast"].clip(lower=1))

    def make_band(frame, col):
        band = {"buckets": BUCKETS, "q10": [], "q90": []}
        for lo, hi in BUCKETS:
            part = frame[(frame["h"] >= lo) & (frame["h"] <= hi)][col]
            band["q10"].append(round(float(part.quantile(0.10)), 4))
            band["q90"].append(round(float(part.quantile(0.90)), 4))
        return band

    

    def inside(frame, band, kind):
        idx = np.digitize(frame["h"], [8, 15, 22])
        q10, q90 = np.array(band["q10"])[idx], np.array(band["q90"])[idx]
        f, scale = frame["forecast"], frame["forecast"].clip(lower=1)
        if kind == "sqrt":
            lower, upper = f + q10 * np.sqrt(scale), f + q90 * np.sqrt(scale)
        else:
            lower, upper = f * (1 + q10), f * (1 + q90)
        return (frame["actual"] >= lower) & (frame["actual"] <= upper)

    

    # Calibrate on the first two start dates, check on the last one
    fit = res[res["origin"] != ORIGINS[-1]]
    check = res[res["origin"] == ORIGINS[-1]].copy()
    check["volume"] = pd.qcut(check["forecast"], 3, labels=["low", "mid", "high"])
    for kind, col in [("relative", "rel"), ("sqrt", "z")]:
        hit = inside(check, make_band(fit, col), kind)
        print(f"\nBand '{kind}': coverage on the held-out start date {hit.mean():.0%} (target 80%)")
        print(hit.groupby(check["volume"], observed=True).mean().round(2).to_string())
    
    # Save both, recalibrated on all start dates
    final = {"relative": make_band(res, "rel"), "sqrt": make_band(res, "z")}
    Path("models/band_v1.json").write_text(json.dumps(final, indent=2))
    res.to_csv("docs/recursive_eval.csv", index=False)


if __name__ == "__main__":
    main()