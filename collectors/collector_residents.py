"""
Stage 1 — Collect resident population data.
Source: Seoul Open Data / 상권분석서비스(상주인구-상권배후지)
Target: raw.residents_raw

Usage: python -m collectors.collector_residents
"""

import sys
import psycopg2.extras
from dotenv import load_dotenv
from collectors.seoul_api import fetch_all_pages
from database.connection import get_conn

load_dotenv()

SERVICE_NAME = "VwsmTrdarWrcPopltnQq"

COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "TOT_WRC_POPLTN_CO": "tot_popltn_co",      # WRC 추가
    "ML_WRC_POPLTN_CO": "ml_popltn_co",         # WRC 추가
    "FML_WRC_POPLTN_CO": "fml_popltn_co",       # WRC 추가
    "AGRDE_10_WRC_POPLTN_CO": "agrde_10_popltn_co",
    "AGRDE_20_WRC_POPLTN_CO": "agrde_20_popltn_co",
    "AGRDE_30_WRC_POPLTN_CO": "agrde_30_popltn_co",
    "AGRDE_40_WRC_POPLTN_CO": "agrde_40_popltn_co",
    "AGRDE_50_WRC_POPLTN_CO": "agrde_50_popltn_co",
    "AGRDE_60_ABOVE_WRC_POPLTN_CO": "agrde_60_above_popltn_co",
    "TOT_HSHLD_CO": "tot_hshld_co",
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
    sql = f"INSERT INTO raw.residents_raw ({', '.join(cols)}) VALUES ({', '.join([f'%({c})s' for c in cols])})"
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.residents_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: Resident Population")
    print("=" * 60)
    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched")
        sys.exit(1)
    load_to_db(rows)


if __name__ == "__main__":
    main()
