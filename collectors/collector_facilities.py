"""
Stage 1 — Collect attraction facility data (external data #3).
Source: Seoul Open Data / 상권분석서비스(집객시설-상권)
Target: raw.facilities_raw

Usage: python -m collectors.collector_facilities
"""

import sys
import psycopg2.extras
from dotenv import load_dotenv
from collectors.seoul_api import fetch_all_pages
from database.connection import get_conn

load_dotenv()

SERVICE_NAME = "VwsmTrdarFcltyQq"

COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "BUS_TRMINL_CO": "bus_trminl_co",
    "SUBWAY_STATN_CO": "subway_statn_co",
    "BUS_STTN_CO": "bus_sttn_co",
    "PBLC_CMCLT_CO": "pblc_cmclt_co",
    "BANK_CO": "bank_co",
    "GNRL_HSPTL_CO": "gnrl_hsptl_co",
    "PHARMCY_CO": "pharmcy_co",
    "KNDRGRT_CO": "kndrgrt_co",
    "ELESCH_CO": "elesch_co",
    "MSKUL_CO": "mskul_co",
    "HGSCHL_CO": "hgschl_co",
    "UNIV_CO": "univ_co",
    "DPRTM_STR_CO": "dprtm_str_co",
    "SUPMK_CO": "supmk_co",
    "THEAT_CO": "theat_co",
    "STAYNG_FCLTY_CO": "stayng_fclty_co",
    "PBLC_PARKNG_LOT_CO": "pblc_parkng_lot_co",
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
    sql = f"INSERT INTO raw.facilities_raw ({', '.join(cols)}) VALUES ({', '.join([f'%({c})s' for c in cols])})"
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
    print(f"  Inserted {len(db_rows)} rows into raw.facilities_raw")
    return len(db_rows)


def main():
    print("=" * 60)
    print("Collector: Attraction Facilities (External Data #3)")
    print("=" * 60)
    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched")
        sys.exit(1)
    load_to_db(rows)


if __name__ == "__main__":
    main()
