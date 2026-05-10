"""
Export key data from PostgreSQL to CSV for Streamlit dashboard.
Run this before deploying to Streamlit Cloud.

Usage: python run_export.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from database.connection import get_conn

load_dotenv()

EXPORT_DIR = "data/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


def export_table(query, filename, desc):
    with get_conn() as conn:
        df = pd.read_sql(query, conn)
    path = os.path.join(EXPORT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  {desc}: {len(df):,} rows -> {path}")
    return df


def main():
    print("Exporting data for dashboard...")

    # 1. Risk features summary by industry
    export_table("""
        SELECT
            svc_induty_cd_nm AS industry,
            COUNT(*) AS obs_count,
            ROUND(AVG(next_q_closure_rate)::NUMERIC, 4) AS avg_closure_rate,
            ROUND(STDDEV(next_q_closure_rate)::NUMERIC, 4) AS std_closure_rate,
            ROUND(AVG(high_risk_flag)::NUMERIC, 4) AS high_risk_pct,
            ROUND(AVG(total_sales)::NUMERIC, 0) AS avg_sales,
            ROUND(AVG(store_count)::NUMERIC, 1) AS avg_store_count,
            ROUND(AVG(franchise_ratio)::NUMERIC, 4) AS avg_franchise_ratio
        FROM mart.risk_features
        GROUP BY svc_induty_cd_nm
        ORDER BY avg_closure_rate DESC
    """, "industry_summary.csv", "Industry summary")

    # 2. Quarterly trend
    export_table("""
        SELECT
            stdr_yyqu_cd AS quarter,
            COUNT(*) AS obs_count,
            ROUND(AVG(next_q_closure_rate)::NUMERIC, 4) AS avg_closure_rate,
            ROUND(AVG(high_risk_flag)::NUMERIC, 4) AS high_risk_pct,
            ROUND(AVG(total_sales)::NUMERIC, 0) AS avg_sales,
            ROUND(AVG(total_foot_traffic)::NUMERIC, 0) AS avg_foot_traffic
        FROM mart.risk_features
        GROUP BY stdr_yyqu_cd
        ORDER BY stdr_yyqu_cd
    """, "quarterly_trend.csv", "Quarterly trend")

    # 3. District-level summary (for map)
    export_table("""
        SELECT
            trdar_cd,
            trdar_cd_nm AS district_name,
            COUNT(*) AS obs_count,
            ROUND(AVG(next_q_closure_rate)::NUMERIC, 4) AS avg_closure_rate,
            ROUND(AVG(high_risk_flag)::NUMERIC, 4) AS high_risk_pct,
            ROUND(AVG(store_count)::NUMERIC, 1) AS avg_store_count,
            ROUND(AVG(total_sales)::NUMERIC, 0) AS avg_sales
        FROM mart.risk_features
        GROUP BY trdar_cd, trdar_cd_nm
        ORDER BY avg_closure_rate DESC
    """, "district_summary.csv", "District summary")

    # 4. Full risk features sample (for interactive exploration)
    export_table("""
        SELECT * FROM mart.risk_features
        WHERE stdr_yyqu_cd >= '20241'
        LIMIT 50000
    """, "risk_features_sample.csv", "Risk features sample (recent)")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
