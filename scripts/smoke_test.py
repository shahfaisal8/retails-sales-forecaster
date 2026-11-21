"""Smoke test for the deployed system: app health, latency, and the daily forecast file in S3."""
import sys
import time
import urllib.request
from datetime import datetime, timezone

import boto3

BUCKET = "faisal-salesforecaster-bucket9056"
MAX_FILE_AGE_HOURS = 36


def check_app(base_url):
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=10) as resp:
            ok = resp.status == 200 and resp.read().decode().strip() == "ok"
        latencies.append((time.perf_counter() - start) * 1000)
        if not ok:
            return False, "health endpoint did not answer ok"
    return True, f"health ok, latency avg {sum(latencies) / 5:.0f} ms, max {max(latencies):.0f} ms"


def check_daily_job():
    s3 = boto3.client("s3", region_name="ap-southeast-2")
    newest = None
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="forecasts/"):
        for obj in page.get("Contents", []):
            if obj["Key"].startswith("forecasts/app/"):
                continue  # files written by the app's batch tab, not by the daily job
            if newest is None or obj["LastModified"] > newest["LastModified"]:
                newest = obj
    if newest is None:
        return False, "no daily forecast files found"
    age_h = (datetime.now(timezone.utc) - newest["LastModified"]).total_seconds() / 3600
    return age_h < MAX_FILE_AGE_HOURS, f"newest daily file {newest['Key']} is {age_h:.1f} hours old"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: py scripts/smoke_test.py http://<app-ip>:8501")
    failed = False
    for name, check in [("app", lambda: check_app(sys.argv[1].rstrip("/"))), ("daily job", check_daily_job)]:
        try:
            ok, message = check()
        except Exception as exc:
            ok, message = False, f"error: {exc}"
        print(("PASS" if ok else "FAIL"), name, "-", message)
        failed = failed or not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()