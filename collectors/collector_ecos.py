"""
Stage 1 — Collect macro economic indicators from ECOS (Bank of Korea).
Source: ECOS API (ecos.bok.or.kr)
Target: raw.ecos_raw
Transferred from: sportswear-brand-monitor/collectors/collector_ecos.py

Usage: python -m collectors.collector_ecos
"""

# stdlib
import os
import sys
import time

# third-party
import requests
import psycopg2.extras
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()

ECOS_API_KEY = os.getenv("ECOS_API_KEY")
ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"
CALL_INTERVAL = 1.0

# Indicators to collect
# CSI: Consumer Sentiment Index
INDICATORS = [
    {
        "stat_code": "511Y002",
        "item_code": "FME",
        "item_code2": "99988",
        "label": "Consumer Sentiment Index (CSI)",
        "start": "202201",
        "end": "202612",
        "cycle": "M",
    },
]


def fetch_ecos(stat_code, item_code, item_code2, start, end, cycle="M"):
    """Fetch data from ECOS API."""
    if not ECOS_API_KEY:
        print("[ERROR] ECOS_API_KEY not set in .env")
        return []

    url = (
        f"{ECOS_BASE_URL}/{ECOS_API_KEY}/json/kr/1/1000/"
        f"{stat_code}/{cycle}/{start}/{end}/{item_code}/{item_code2}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] ECOS request failed: {e}")
        return []

    if "StatisticSearch" not in data:
        if "RESULT" in data:
            print(f"  [ERROR] {data['RESULT'].get('MESSAGE', '')}")
        return []

    rows = data["StatisticSearch"].get("row", [])
    return rows


def load_to_db(rows):
    """Insert ECOS rows into raw.ecos_raw."""
    if not rows:
        return 0

    db_rows = []
    for r in rows:
        db_rows.append({
            "stat_code": r.get("STAT_CODE"),
            "stat_name": r.get("STAT_NAME"),
            "item_code": r.get("ITEM_CODE1"),
            "item_name": r.get("ITEM_NAME1"),
            "time_code": r.get("TIME"),
            "value": r.get("DATA_VALUE"),
        })

    cols = ["stat_code", "stat_name", "item_code", "item_name", "time_code", "value"]
    col_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = f"INSERT INTO raw.ecos_raw ({col_str}) VALUES ({placeholders})"

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.ecos_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: ECOS Macro Indicators")
    print("=" * 60)

    total = 0
    for ind in INDICATORS:
        print(f"\n  Fetching: {ind['label']}")
        rows = fetch_ecos(
            ind["stat_code"], ind["item_code"], ind["item_code2"],
            ind["start"], ind["end"], ind["cycle"]
        )
        if rows:
            count = load_to_db(rows)
            total += count
        else:
            print(f"  [WARN] No data for {ind['label']}")
        time.sleep(CALL_INTERVAL)

    print(f"\nDone. Total: {total} rows")


if __name__ == "__main__":
    main()
