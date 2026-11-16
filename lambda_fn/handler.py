import json
import logging
import os
import time
from datetime import datetime, timezone
import tempfile

import boto3
import joblib
import pandas as pd

from src.forecast import forecast_next_day

BUCKET = os.environ.get("S3_BUCKET", "faisal-salesforecaster-bucket9056")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/retail_forecaster/v1/model.joblib")
DATA_KEY = os.environ.get("DATA_KEY", "data/latest/sales.csv")
OUTPUT_PREFIX = "forecasts"
REQUIRED = ["date", "product_id", "store_id", "sales_qty"]
MODEL_PATH = os.path.join(tempfile.gettempdir(), "model.joblib")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
_model = None  # kept between runs while the container stays warm


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            s3.download_file(BUCKET, MODEL_KEY, MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


def read_latest_sales():
    body = s3.get_object(Bucket=BUCKET, Key=DATA_KEY)["Body"]
    df = pd.read_csv(body)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Sales data is missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    if df.empty:
        raise ValueError("Sales data has no valid rows")
    return df


def lambda_handler(event, context):
    started = time.perf_counter()
    run_time = datetime.now(timezone.utc)
    try:
        model = load_model()
        df = read_latest_sales()
        out = forecast_next_day(model, df)
        key = f"{OUTPUT_PREFIX}/forecast_{run_time:%Y%m%d}.csv"
        s3.put_object(Bucket=BUCKET, Key=key, Body=out.to_csv(index=False).encode(), ContentType="text/csv")
    except Exception:
        logger.exception("Daily forecast failed")
        raise

    summary = {
        "event": "forecast_written",
        "rows": int(len(out)),
        "forecast_date": str(out["date"].iloc[0].date()),
        "key": key,
        "duration_ms": round((time.perf_counter() - started) * 1000),
    }
    logger.info(json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(lambda_handler({}, None))