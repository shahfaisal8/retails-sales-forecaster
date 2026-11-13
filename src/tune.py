import json
from itertools import product

import pandas as pd

from src.train import FEATURES, TARGET, build_forest, load_splits, mae, mape, wape


def main():
    train, val, _ = load_splits()
    y_val = val[TARGET].to_numpy()

    rows = []
    for depth, leaf in product([12, 16, 20], [5, 20]):
        params = dict(max_depth=depth, min_samples_leaf=leaf, max_samples=0.5)
        pipe = build_forest(**params)
        pipe.fit(train[FEATURES], train[TARGET])
        pred = pipe.predict(val[FEATURES])
        rows.append({**params, "MAE": mae(y_val, pred), "MAPE_%": mape(y_val, pred), "WAPE_%": wape(y_val, pred)})
        print(rows[-1])

    results = pd.DataFrame(rows).sort_values("MAE").round(3)
    results.to_csv("docs/tuning_results.csv", index=False)
    best = results.iloc[0][["max_depth", "min_samples_leaf", "max_samples"]].to_dict()
    best = {"max_depth": int(best["max_depth"]), "min_samples_leaf": int(best["min_samples_leaf"]), "max_samples": float(best["max_samples"])}
    with open("docs/best_params.json", "w") as f:
        json.dump(best, f)
    print("\nBest settings:", best)


if __name__ == "__main__":
    main()