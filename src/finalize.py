import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.train import FEATURES, TARGET, build_forest, build_linear, load_splits, report

MODEL_DIR = Path("models")
DEPLOY_PARAMS = dict(n_estimators=100, max_depth=12, min_samples_leaf=20, max_samples=0.5)


def main():
    train, val, test = load_splits()
    fit_df = pd.concat([train, val])
    y = test[TARGET].to_numpy()

    # 1. Honest evaluation: fit on train + val, score once on the untouched test set
    rows = [
        report("naive (lag_1)", y, test["lag_1"].to_numpy()),
        report("seasonal naive (lag_7)", y, test["lag_7"].to_numpy()),
    ]
    linear = build_linear().fit(fit_df[FEATURES], fit_df[TARGET])
    rows.append(report("linear regression", y, linear.predict(test[FEATURES])))
    forest = build_forest(**DEPLOY_PARAMS).fit(fit_df[FEATURES], fit_df[TARGET])
    rows.append(report("random forest", y, forest.predict(test[FEATURES])))

    results = pd.DataFrame(rows).round(2)
    print(results.to_string(index=False))
    results.to_csv("docs/results_test.csv", index=False)

    # 2. Production model: refit on ALL data so it knows the most recent months
    all_df = pd.concat([fit_df, test.dropna(subset=FEATURES)])
    final = build_forest(**DEPLOY_PARAMS).fit(all_df[FEATURES], all_df[TARGET])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_DIR / "retail_forecaster_v1.joblib", compress=3)
    meta = {
        "version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "RandomForestRegressor",
        "params": DEPLOY_PARAMS,
        "features": FEATURES,
        "trained_on": f"{all_df['date'].min().date()} to {all_df['date'].max().date()}",
        "test_metrics_from_evaluation_fit": results[results["model"] == "random forest"].iloc[0].to_dict(),
    }
    (MODEL_DIR / "retail_forecaster_v1.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved model and metadata to", MODEL_DIR)


if __name__ == "__main__":
    main()