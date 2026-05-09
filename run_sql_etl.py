"""
Stage 2 — Execute SQL ETL pipeline (raw → staging → mart).
Runs SQL files in order to build staging and mart tables.

Usage: python run_sql_etl.py
"""

# stdlib
import os
import time

# third-party
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()

SQL_DIR = "sql"

SQL_FILES = [
    "01_staging_sales.sql",
    "02_staging_stores.sql",
    "03_staging_environment.sql",
    "04_mart_risk_features.sql",
]


def run_sql_file(filepath):
    """Execute a single SQL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print(f"  OK: {os.path.basename(filepath)}")


def verify_tables():
    """Print row counts for all staging/mart tables."""
    tables = [
        ("staging.sales_quarterly", "staging"),
        ("staging.stores_quarterly", "staging"),
        ("staging.environment_quarterly", "staging"),
        ("mart.risk_features", "mart"),
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            print("\n  Table Row Counts:")
            for table, schema in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"    {table}: {count:,}")
                except Exception as e:
                    print(f"    {table}: ERROR - {e}")
                    conn.rollback()


def main():
    print("=" * 60)
    print("Stage 2: SQL ETL Pipeline (raw -> staging -> mart)")
    print("=" * 60)

    for sql_file in SQL_FILES:
        filepath = os.path.join(SQL_DIR, sql_file)
        if not os.path.exists(filepath):
            print(f"  [ERROR] File not found: {filepath}")
            continue

        print(f"\n  Running: {sql_file}")
        start = time.time()
        try:
            run_sql_file(filepath)
            elapsed = time.time() - start
            print(f"  Elapsed: {elapsed:.1f}s")
        except Exception as e:
            print(f"  [ERROR] {sql_file}: {e}")
            return

    verify_tables()

    # Summary stats for risk_features
    print("\n  Risk Features Summary:")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT stdr_yyqu_cd) as quarters,
                    COUNT(DISTINCT trdar_cd) as districts,
                    COUNT(DISTINCT svc_induty_cd) as industries,
                    ROUND(AVG(next_q_closure_rate)::NUMERIC, 4) as avg_closure_rate,
                    ROUND(SUM(high_risk_flag)::NUMERIC / COUNT(*), 4) as high_risk_pct
                FROM mart.risk_features
            """)
            row = cur.fetchone()
            print(f"    Total rows: {row[0]:,}")
            print(f"    Quarters: {row[1]}")
            print(f"    Districts: {row[2]}")
            print(f"    Industries: {row[3]}")
            print(f"    Avg closure rate: {row[4]}")
            print(f"    High risk pct: {row[5]}")

    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
