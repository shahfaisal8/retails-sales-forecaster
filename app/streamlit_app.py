import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_prep import fill_missing_dates
from src.forecast import forecast_next_day, forecast_series

BUCKET = os.environ.get("S3_BUCKET", "faisal-salesforecaster-bucket9056")
DATA_KEY = "data/latest/sales.csv"
MODEL_KEY = "models/retail_forecaster/v1/model.joblib"
META_KEY = "models/retail_forecaster/v1/metadata.json"
MODEL_LOCAL = Path("models/retail_forecaster_v1.joblib")
META_LOCAL = Path("models/retail_forecaster_v1.json")
REQUIRED = ["date", "product_id", "store_id", "sales_qty"]


def get_logger():
    logger = logging.getLogger("forecaster")
    if not logger.handlers:
        Path("logs").mkdir(exist_ok=True)
        handler = logging.FileHandler("logs/app.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


@st.cache_resource
def s3_client():
    return boto3.client("s3")


@st.cache_resource
def load_model():
    if not MODEL_LOCAL.exists():
        MODEL_LOCAL.parent.mkdir(exist_ok=True)
        s3_client().download_file(BUCKET, MODEL_KEY, str(MODEL_LOCAL))
        s3_client().download_file(BUCKET, META_KEY, str(META_LOCAL))
        log.info("Model downloaded from S3")
    return joblib.load(MODEL_LOCAL), json.loads(META_LOCAL.read_text())


@st.cache_data(show_spinner="Fetching data from S3...")
def fetch_from_s3():
    body = s3_client().get_object(Bucket=BUCKET, Key=DATA_KEY)["Body"]
    return pd.read_csv(body)


def validate(raw):
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Expected: {REQUIRED}")
    df = raw[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise ValueError("No valid rows: check the date and sales_qty columns")
    if (df["sales_qty"] < 0).any():
        raise ValueError("Negative sales quantities found")
    return df.drop_duplicates(["date", "store_id", "product_id"])


def id_number(value):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else 0


st.set_page_config(page_title="Cloud Retail Sales Forecaster", layout="wide")
st.title("Cloud Retail Sales Forecaster")

try:
    model, meta = load_model()
except Exception:
    log.exception("Could not load model")
    st.error("The model could not be loaded. Check the logs and your AWS access.")
    st.stop()

source = st.sidebar.radio("Data source", ["Upload CSV", "Fetch from S3"])
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 30)

try:
    if source == "Upload CSV":
        file = st.sidebar.file_uploader("Sales CSV", type="csv")
        if file is None:
            st.info("Upload a CSV with columns: date, product_id, store_id, sales_qty")
            st.stop()
        data = validate(pd.read_csv(file))
    else:
        data = validate(fetch_from_s3())
    log.info("Data loaded: source=%s rows=%d", source, len(data))
except Exception as exc:
    log.exception("Data loading failed")
    st.error(f"Could not read the data: {exc}")
    st.stop()

tab_forecast, tab_batch, tab_model = st.tabs(["Forecast", "Batch (all series)", "About the model"])

with tab_forecast:
    store = st.selectbox("Store", sorted(data["store_id"].unique()))
    products = sorted(data.loc[data["store_id"] == store, "product_id"].unique())
    product = st.selectbox("Product", products)

    series = data[(data["store_id"] == store) & (data["product_id"] == product)]
    filled = fill_missing_dates(series)
    if len(filled) < 30:
        st.warning("Not enough history; forecasts may be unreliable")

    try:
        started = time.perf_counter()
        fc, _ = forecast_series(model, filled, id_number(store), id_number(product), horizon)
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info("Forecast: store=%s product=%s horizon=%d days_history=%d ms=%.0f",
                 store, product, horizon, len(filled), elapsed_ms)
    except Exception as exc:
        log.exception("Forecast failed")
        st.error(f"Forecast failed: {exc}")
        st.stop()

    fig, ax = plt.subplots(figsize=(11, 4))
    recent = filled.tail(120)
    ax.plot(recent["date"], recent["sales_qty"], label="History")
    ax.plot(fc["date"], fc["forecast"], label="Forecast")
    ax.fill_between(fc["date"], fc["lower"], fc["upper"], alpha=0.25, label="Model spread (not a prediction interval)")
    ax.set(title=f"{store} / {product}: history and {horizon}-day forecast", ylabel="Units")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    st.caption(f"Computed in {elapsed_ms:.0f} ms. The band shows how much the model's trees disagree; "
               "it understates real uncertainty, especially further out.")

    st.download_button("Download forecast CSV", fc.to_csv(index=False), f"forecast_{store}_{product}.csv")

    test_metrics = meta["test_metrics_from_evaluation_fit"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE on test (units)", test_metrics["MAE"])
    c2.metric("MAPE on test (%)", test_metrics["MAPE_%"])
    c3.metric("WAPE on test (%)", test_metrics["WAPE_%"])
    st.caption("Test metrics are next-day accuracy over all 500 series, from the held-out test period.")

with tab_batch:
    st.write("Next-day forecast for every store-product series in the loaded data.")
    if st.button("Run batch forecast and save to S3"):
        try:
            out = forecast_next_day(model, data)
            key = f"forecasts/app/forecast_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.csv"
            s3_client().put_object(Bucket=BUCKET, Key=key, Body=out.to_csv(index=False).encode())
            log.info("Batch forecast saved: s3://%s/%s rows=%d", BUCKET, key, len(out))
            st.success(f"Saved {len(out)} forecasts to s3://{BUCKET}/{key}")
            st.dataframe(out.head(20))
        except Exception as exc:
            log.exception("Batch forecast failed")
            st.error(f"Batch forecast failed: {exc}")

with tab_model:
    st.json(meta)