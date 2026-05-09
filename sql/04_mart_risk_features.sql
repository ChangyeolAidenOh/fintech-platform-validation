-- ================================================================
-- Stage 2: mart.risk_features
-- Core analysis table for Model A/B/C ablation study
-- Grain: stdr_yyqu_cd × trdar_cd × svc_induty_cd
-- Joins: sales (card features) + stores (target) + environment (external) + ECOS (macro)
-- ================================================================

DROP TABLE IF EXISTS mart.risk_features CASCADE;

CREATE TABLE mart.risk_features AS
WITH macro AS (
    SELECT
        LEFT(time_code, 4) || CASE
            WHEN RIGHT(time_code, 2) IN ('01','02','03') THEN '1'
            WHEN RIGHT(time_code, 2) IN ('04','05','06') THEN '2'
            WHEN RIGHT(time_code, 2) IN ('07','08','09') THEN '3'
            WHEN RIGHT(time_code, 2) IN ('10','11','12') THEN '4'
        END AS stdr_yyqu_cd,
        AVG(value) AS csi_avg
    FROM raw.ecos_raw
    WHERE stat_code = '511Y002'
    GROUP BY 1
),
combined AS (
    SELECT
        s.stdr_yyqu_cd,
        s.trdar_cd,
        s.trdar_cd_nm,
        s.svc_induty_cd,
        s.svc_induty_cd_nm,

        -- GROUP A: Card sales features (Model A input)
        s.total_sales,
        s.total_txn_count,
        s.avg_sales_per_txn,
        s.weekend_sales_ratio,
        s.female_sales_ratio,
        s.young_adult_sales_ratio,
        s.sales_qoq_change,
        s.txn_count_qoq_change,
        s.sales_vs_2q_ma,

        -- GROUP B: External features (added in Model B)
        e.total_foot_traffic,
        e.traffic_female_ratio,
        e.traffic_young_adult_ratio,
        e.traffic_qoq_change,
        e.change_index_code,
        e.change_index_name,
        e.change_index_numeric,
        e.operating_months_avg,
        e.closed_months_avg,
        e.subway_count,
        e.bus_stop_count,
        e.total_facility_count,
        e.total_residents,
        e.total_households,
        e.resident_young_adult_ratio,
        e.resident_elderly_ratio,
        m.csi_avg,

        -- Store context features
        st.store_count,
        st.open_count,
        st.close_count,
        st.net_store_change,
        st.franchise_ratio,
        st.closure_rate,
        st.competition_density,
        st.store_count_qoq_change,

        -- TARGET VARIABLE
        st.next_q_closure_rate

    FROM staging.sales_quarterly s
    INNER JOIN staging.stores_quarterly st
        ON s.stdr_yyqu_cd = st.stdr_yyqu_cd
        AND s.trdar_cd = st.trdar_cd
        AND s.svc_induty_cd = st.svc_induty_cd
    LEFT JOIN staging.environment_quarterly e
        ON s.stdr_yyqu_cd = e.stdr_yyqu_cd
        AND s.trdar_cd = e.trdar_cd
    LEFT JOIN macro m
        ON s.stdr_yyqu_cd = m.stdr_yyqu_cd
    WHERE st.next_q_closure_rate IS NOT NULL
),
industry_median AS (
    SELECT
        svc_induty_cd,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY next_q_closure_rate) AS median_closure
    FROM combined
    GROUP BY svc_induty_cd
)
SELECT
    c.*,
    CASE WHEN c.next_q_closure_rate > m.median_closure * 1.5
         THEN 1 ELSE 0
    END AS high_risk_flag
FROM combined c
LEFT JOIN industry_median m ON c.svc_induty_cd = m.svc_induty_cd;

-- Indexes
CREATE INDEX idx_risk_features_keys
    ON mart.risk_features(stdr_yyqu_cd, trdar_cd, svc_induty_cd);
CREATE INDEX idx_risk_features_industry
    ON mart.risk_features(svc_induty_cd);
