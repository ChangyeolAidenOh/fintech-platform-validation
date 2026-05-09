"""
Stage 1 — Collect store data (open/close counts).
Source: Seoul Open Data / 상권분석서비스(점포-상권)
Target: raw.stores_raw
This is the TARGET VARIABLE source — 폐업수/폐업률 from here.

Usage: python -m collectors.collector_stores
"""

# stdlib
import sys

# third-party
import psycopg2.extras
from dotenv import load_dotenv

# local
from collectors.seoul_api import fetch_all_pages
from database.connection import get_conn

load_dotenv()

SERVICE_NAME = "VwsmTrdarStorQq"

COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "SVC_INDUTY_CD": "svc_induty_cd",
    "SVC_INDUTY_CD_NM": "svc_induty_cd_nm",
    "STOR_CO": "stor_co",
    "SIMILR_INDUTY_STOR_CO": "similr_induty_stor_co",
    "OPBIZ_RT": "opbiz_rt",
    "OPBIZ_STOR_CO": "opbiz_stor_co",
    "CLSBIZ_RT": "clsbiz_rt",
    "CLSBIZ_STOR_CO": "clsbiz_stor_co",
    "FRC_STOR_CO": "frc_stor_co",
}


def transform_row(api_row):
    """Map API response keys to DB column values."""
    result = {}
    for api_key, db_col in COLUMN_MAP.items():
        val = api_row.get(api_key)
        result[db_col] = val if val is not None and val != "" else None
    return result


def load_to_db(rows):
    """Insert rows into raw.stores_raw."""
    if not rows:
        return 0
    db_rows = [transform_row(r) for r in rows]
    cols = list(COLUMN_MAP.values())
    col_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = f"INSERT INTO raw.stores_raw ({col_str}) VALUES ({placeholders})"

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.stores_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: Store Open/Close Data (Target Variable)")
    print("=" * 60)
    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched")
        sys.exit(1)
    count = load_to_db(rows)
    print(f"Done. Total: {count} rows")


if __name__ == "__main__":
    main()
