# Progress notes: Cloud Retail Sales Forecaster (TCS ION Industry Project)

## Status (21 Nov 2026)
Core system built and deployed. Left: smoke tests and screenshots, README,
short report, 2-3 min demo video, fixed URL (Elastic IP), final clean-up.

## What exists
- Data: Kaggle Store Item Demand Forecasting, 500 store-product series (2013-2017),
  split by time 70/15/15 (src/data_prep.py)
- Features: lags 1/7/14/28, moving averages 7/14, calendar features (src/features.py)
- Test results (MAE / WAPE %): seasonal naive 9.21 / 14.61, linear 7.00 / 11.10,
  random forest 6.71 / 10.64
- Deployed model v1: random forest (100 trees, depth 12, leaf 20), 14.6 MB
- 3-fold backtest, tuning, rejected day-of-year experiment: results in docs/
- 30-day recursive forecast: MAE 6.98 vs 9.71 seasonal naive; calibrated 80% band
  (square-root scaled) from past errors
- 7 pytest tests pass: py -m pytest -q
- Streamlit app (app/streamlit_app.py) on EC2, systemd service "forecaster", port 8501
- Lambda retail-forecaster-daily (Python 3.14, x86_64) + EventBridge
  cron(30 20 * * ? *) = 02:00 IST, writes forecasts/forecast_YYYYMMDD.csv

## AWS names (region ap-southeast-2)
Bucket faisal-salesforecaster-bucket9056, roles forecaster-lambda-role and
forecaster-ec2-role, instance retail-forecaster-app, security group forecaster-app-sg.

## How to run
- Always from the repo root, with -m: py -m src.train, py -m pytest -q
- App locally: py -m streamlit run app/streamlit_app.py
- Lambda package: py scripts\build_lambda_zip.py, then upload to S3 and update the function
- Deploy app changes: on the server, git pull && sudo systemctl restart forecaster

## After stopping the instance
Start it in the EC2 console; the app starts by itself. Public IP changes unless an
Elastic IP is attached.

## Never commit
AWS access key CSV files, .env, .pem files. (A key was committed once and rotated.)

## Next
Smoke tests with screenshots, README, report (data, model, metrics, system design,
costs), demo video, Elastic IP, clean-up checklist at the very end