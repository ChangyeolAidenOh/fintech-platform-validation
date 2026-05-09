"""
Stage 1 — Collect district change indicators (external data #2).
Source: Seoul Open Data / 상권분석서비스(상권변화지표-상권)
Target: raw.district_change_raw

Usage: python -m collectors.collector_change
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

SERVICE_NAME = "VwsmTrdarIxQq"

COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "TRDAR_CHNGE_IX": "trdar_chg_ind",
    "TRDAR_CHNGE_IX_NM": "trdar_chg_ind_nm",
    "OPR_SALE_MT_AVRG": "opbiz_rt",
    "CLS_SALE_MT_AVRG": "clsbiz_rt",
}


def transform_row(api_row):
    result = {}
    for api_key, db_col in COLUMN_MAP.items():
        val = api_row.get(api_key)
        result[db_col] = val if val is not None and val != "" else None
    return result


def load_to_db(rows):
    if not rows:
        return 0
    db_rows = [transform_row(r) for r in rows]
    cols = list(COLUMN_MAP.values())
    col_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])
    sql = f"INSERT INTO raw.district_change_raw ({col_str}) VALUES ({placeholders})"
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.district_change_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: District Change Indicator (External Data #2)")
    print("=" * 60)
    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched")
        sys.exit(1)
    load_to_db(rows)


if __name__ == "__main__":
    main()
