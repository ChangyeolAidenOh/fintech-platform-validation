"""
Database connection utility.
Provides get_conn() context manager for PostgreSQL access.
Mirrors sportswear-brand-monitor/database/connection.py pattern.
"""

# stdlib
import os
from contextlib import contextmanager

# third-party
import psycopg2
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "bc_risk")
POSTGRES_USER = os.getenv("POSTGRES_USER", "bc_risk_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "bc_risk_pass")


@contextmanager
def get_conn():
    """Yield a psycopg2 connection; auto-commit on success, rollback on error."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
