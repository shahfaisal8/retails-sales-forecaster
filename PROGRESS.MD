# Project Progress — Retail Sales Forecaster

## Done
**Data & Features:** Cleaned the Kaggle store-item dataset, filled missing dates, split chronologically (train/val/test), and built lag (`1, 7, 14, 28`), rolling mean (`7, 14`), weekend, and calendar/ID features.
**Models & Evaluation:** Benchmarked baselines, Linear Regression, and Random Forest across validation, a 3-fold rolling-origin backtest, and the held-out 2017 test set (6.71 MAE / 10.64% WAPE). Rejected the `dayofyear` feature experiment (~0.6% gain, below the 2% threshold) and uploaded `retail_forecaster_v1` (~14.6 MB) + metadata to S3.
**Forecasting Engine & Tests:** Built `src/forecast.py` (30-day recursive forecasting + daily batch inference) and verified 6 `pytest` unit tests covering training-serving skew, short history, flat demand, weekly patterns, missing dates, and empty-history errors.
**Calibrated Band:** Upgraded the recursive prediction interval using a square-root error scaling approach.
**Streamlit App (Local):** Built and tested `app/streamlit_app.py` locally (CSV upload & S3 fetch, recursive forecast chart, test metrics, batch-to-S3 export, and `logs/app.log` logging).
**Lambda Deployed & Tested:** Packaged and deployed the daily batch forecasting Lambda function (`retail-forecaster-daily`) and verified it against S3.

## Key Decisions**Production Model:** Random Forest (`n_estimators=100`, `max_depth=12`, `min_samples_leaf=20`, `max_samples=0.5`) chosen for top accuracy and compact ~14.6 MB artifact size.
**Uncertainty Band:** Square-root band scaling to account for uncertainty growth across the 30-day recursive horizon *(confirm exact multiplier/calibration details from your recent chat if needed)*.
**Cloud & Runtime Setup:** AWS Sydney region (`ap-southeast-2`), Lambda built via Python 3.14 zip package targeting `x86_64` architecture.

## Next
- Verify EventBridge schedule trigger for the daily Lambda run
- Run failure smoke tests (e.g., bad/missing inputs, permissions/error paths)
- Deploy the Streamlit app on EC2
- Finalize `README.md`
- Write the project report (including architecture, trade-offs, and AWS cost breakdown)
- Record the demo video

## AWS Resource Names
- **S3 Bucket:** `faisal-salesforecaster-bucket9056`
- **Lambda Function:** `retail-forecaster-daily`
- **IAM Role:** `forecaster-lambda-role`