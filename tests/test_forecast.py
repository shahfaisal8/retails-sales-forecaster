from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_prep import fill_missing_dates
from src.features import add_features
from src.forecast import feature_row, forecast_series
from src.train import FEATURES

MODEL_PATH = Path("models/retail_forecaster_v1.joblib")


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


def make_series(values, start="2017-01-02"):
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "sales_qty": values})


def test_inference_features_match_training_features():
    rng = np.random.default_rng(0)
    s = make_series(rng.integers(20, 80, size=60).astype(float))
    s["store_id"], s["product_id"] = "S001", "P001"
    trained = add_features(s).iloc[-1]
    row = feature_row(s["sales_qty"].tolist()[:-1], s["date"].iloc[-1], 1, 1)
    for col in FEATURES:
        assert row[col] == pytest.approx(trained[col])


def test_short_series_warns_but_still_forecasts(model):
    s = make_series([40, 42, 38, 45, 50, 60, 55, 41, 43, 39])
    out, warning = forecast_series(model, s, 1, 1, horizon=30)
    assert len(out) == 30
    assert warning is not None
    assert (out["forecast"] >= 0).all()


def test_flat_demand_forecast_stays_near_the_mean(model):
    s = make_series([50.0] * 120)
    out, _ = forecast_series(model, s, 1, 1, horizon=14)
    assert abs(out["forecast"].mean() - 50) < 8


def test_weekly_pattern_is_followed(model):
    pattern = np.array([40, 42, 45, 50, 60, 75, 80], dtype=float)  # Monday..Sunday
    dates = pd.date_range("2017-01-02", periods=140, freq="D")  # starts on a Monday
    s = pd.DataFrame({"date": dates, "sales_qty": pattern[dates.dayofweek]})
    out, _ = forecast_series(model, s, 1, 1, horizon=14)
    expected = pattern[out["date"].dt.dayofweek]
    assert np.corrcoef(out["forecast"], expected)[0, 1] > 0.8


def test_missing_dates_are_filled():
    s = make_series([float(i) for i in range(1, 11)])
    s = s.drop(index=[3, 4]).reset_index(drop=True)
    s["store_id"], s["product_id"] = "S001", "P001"
    filled = fill_missing_dates(s, method="zero")
    assert len(filled) == 10
    assert filled["sales_qty"].isna().sum() == 0
    assert (filled["sales_qty"] == 0).sum() == 2


def test_empty_history_raises_clear_error(model):
    with pytest.raises(ValueError):
        forecast_series(model, make_series([]), 1, 1)