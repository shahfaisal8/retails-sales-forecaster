import json
from pathlib import Path

import numpy as np
import pandas as pd

from train import FEATURES, TARGET, build_forest, build_linear, load_splits, mae, mape, wape


def load_best_params():
    path = Path("docs/best_params.json")
    return json.loads(path.read_text()) if path.exists() else {}


def rolling_folds(df, fractions=(0.55, 0.70, 0.85), horizon=90):
    dates = np.sort(df["date"].unique())
    n = len(dates)
    for f in fractions:
        i = int(n * f)
        origin = dates[i]
        end = dates[min(i + horizon, n) - 1]
        train = df[df["date"] < origin].dropna(subset=FEATURES)
        test = df[(df["date"] >= origin) & (df["date"] <= end)]
        yield origin, end, train, test


def main():
    train, val, _ = load_splits()  # the test set is never used here
    df = pd.concat([train, val])
    best = load_best_params()

    rows = []
    for k, (origin, end, tr, te) in enumerate(rolling_folds(df), start=1):
        y = te[TARGET].to_numpy()
        models = {
            "seasonal naive": te["lag_7"].to_numpy(),
            "linear regression": build_linear().fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES]),
            "random forest": build_forest(**best).fit(tr[FEATURES], tr[TARGET]).predict(te[FEATURES]),
        }
        for name, pred in models.items():
            rows.append({"fold": k, "from": str(origin)[:10], "to": str(end)[:10], "model": name,
                         "MAE": mae(y, pred), "MAPE_%": mape(y, pred), "WAPE_%": wape(y, pred)})
        print("finished fold", k, str(origin)[:10], "->", str(end)[:10])

    res = pd.DataFrame(rows).round(2)
    res.to_csv("docs/backtest_results.csv", index=False)
    print(res.to_string(index=False))
    print("\nAverage over folds:")
    print(res.groupby("model")[["MAE", "MAPE_%", "WAPE_%"]].mean().round(2))


if __name__ == "__main__":
    main()