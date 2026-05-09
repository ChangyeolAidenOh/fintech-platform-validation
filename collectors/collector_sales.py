"""
Stage 1 — Collect estimated sales data (card-transaction based).
Source: Seoul Open Data / 상권분석서비스(추정매출-상권)
Target: raw.sales_raw

Usage: python -m collectors.collector_sales
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

# Service name — update if spike_api_check.py finds a different name
SERVICE_NAME = "VwsmTrdarSelngQq"

# Column mapping: API response key -> DB column
COLUMN_MAP = {
    "STDR_YYQU_CD": "stdr_yyqu_cd",
    "TRDAR_CD": "trdar_cd",
    "TRDAR_CD_NM": "trdar_cd_nm",
    "SVC_INDUTY_CD": "svc_induty_cd",
    "SVC_INDUTY_CD_NM": "svc_induty_cd_nm",
    "THSMON_SELNG_AMT": "thsmon_selng_amt",
    "THSMON_SELNG_CO": "thsmon_selng_co",
    "MDWK_SELNG_AMT": "mdwk_selng_amt",
    "WKEND_SELNG_AMT": "wkend_selng_amt",
    "ML_SELNG_AMT": "ml_selng_amt",
    "FML_SELNG_AMT": "fml_selng_amt",
    "AGRDE_10_SELNG_AMT": "agrde_10_selng_amt",
    "AGRDE_20_SELNG_AMT": "agrde_20_selng_amt",
    "AGRDE_30_SELNG_AMT": "agrde_30_selng_amt",
    "AGRDE_40_SELNG_AMT": "agrde_40_selng_amt",
    "AGRDE_50_SELNG_AMT": "agrde_50_selng_amt",
    "AGRDE_60_ABOVE_SELNG_AMT": "agrde_60_above_selng_amt",
}

INSERT_SQL = """
    INSERT INTO raw.sales_raw ({cols})
    VALUES ({placeholders})
    ON CONFLICT DO NOTHING
"""


def transform_row(api_row):
    """Map API response keys to DB column values."""
    result = {}
    for api_key, db_col in COLUMN_MAP.items():
        val = api_row.get(api_key)
        if val is not None and val != "":
            result[db_col] = val
        else:
            result[db_col] = None
    return result


def load_to_db(rows):
    """Insert rows into raw.sales_raw."""
    if not rows:
        print("No rows to insert")
        return 0

    db_rows = [transform_row(r) for r in rows]
    cols = list(COLUMN_MAP.values())
    col_str = ", ".join(cols)
    placeholders = ", ".join([f"%({c})s" for c in cols])

    sql = f"""
        INSERT INTO raw.sales_raw ({col_str})
        VALUES ({placeholders})
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, db_rows, page_size=500)
            count = len(db_rows)
    print(f"  Inserted {count} rows into raw.sales_raw")
    return count


def main():
    print("=" * 60)
    print("Collector: Estimated Sales (Card-Transaction Based)")
    print("=" * 60)

    rows = fetch_all_pages(SERVICE_NAME)
    if not rows:
        print("[ERROR] No data fetched. Check API key and service name.")
        sys.exit(1)

    count = load_to_db(rows)
    print(f"Done. Total: {count} rows")


if __name__ == "__main__":
    main()
