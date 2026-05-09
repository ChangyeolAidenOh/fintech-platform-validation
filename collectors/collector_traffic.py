"""
Stage 1 — Collect foot traffic data (external data #1).
Source: Seoul Open Data / 상권분석서비스(길단위인구-상권배후지)
Target: raw.foot_traffic_raw

Usage: python -m collectors.collector_traffic
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

SERVICE_NAME = "VwsmTrdarFlpopQq"

COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "TOT_FLPOP_CO": "tot_flpop_co",
    "ML_FLPOP_CO": "ml_flpop_co",
    "FML_FLPOP_CO": "fml_flpop_co",
    "AGRDE_10_FLPOP_CO": "agrde_10_flpop_co",
    "AGRDE_20_FLPOP_CO": "agrde_20_flpop_co",
    "AGRDE_30_FLPOP_CO": "agrde_30_flpop_co",
    "AGRDE_40_FLPOP_CO": "agrde_40_flpop_co",
    "AGRDE_50_FLPOP_CO": "agrde_50_flpop_co",
    "AGRDE_60_ABOVE_FLPOP_CO": "agrde_60_above_flpop_co",
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
    sql = f"INSERT INTO raw.foot_traffic_raw ({col_str}) VALUES ({placeholders})"
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.foot_traffic_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: Foot Traffic (External Data #1)")
    print("=" * 60)
    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched")
        sys.exit(1)
    load_to_db(rows)


if __name__ == "__main__":
    main()
