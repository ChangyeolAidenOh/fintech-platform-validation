-- ================================================================
-- Stage 2: staging.sales_quarterly
-- Source: raw.sales_raw
-- Grain: stdr_yyqu_cd × trdar_cd × svc_induty_cd
-- Features: sales levels, composition ratios, QoQ changes
-- ================================================================

DROP TABLE IF EXISTS staging.sales_quarterly CASCADE;

CREATE TABLE staging.sales_quarterly AS
WITH base AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        trdar_cd_nm,
        svc_induty_cd,
        svc_induty_cd_nm,
        COALESCE(thsmon_selng_amt, 0)   AS total_sales,
        COALESCE(thsmon_selng_co, 0)    AS total_txn_count,
        COALESCE(mdwk_selng_amt, 0)     AS weekday_sales,
        COALESCE(wkend_selng_amt, 0)    AS weekend_sales,
        COALESCE(ml_selng_amt, 0)       AS male_sales,
        COALESCE(fml_selng_amt, 0)      AS female_sales,
        COALESCE(agrde_10_selng_amt, 0) AS age_10_sales,
        COALESCE(agrde_20_selng_amt, 0) AS age_20_sales,
        COALESCE(agrde_30_selng_amt, 0) AS age_30_sales,
        COALESCE(agrde_40_selng_amt, 0) AS age_40_sales,
        COALESCE(agrde_50_selng_amt, 0) AS age_50_sales,
        COALESCE(agrde_60_above_selng_amt, 0) AS age_60_plus_sales
    FROM raw.sales_raw
),
with_derived AS (
    SELECT
        *,
        -- Per-transaction average
        CASE WHEN total_txn_count > 0
             THEN total_sales / total_txn_count
             ELSE NULL
        END AS avg_sales_per_txn,

        -- Composition ratios (nullsafe)
        CASE WHEN total_sales > 0
             THEN weekend_sales::NUMERIC / total_sales
             ELSE NULL
        END AS weekend_sales_ratio,

        CASE WHEN total_sales > 0
             THEN female_sales::NUMERIC / total_sales
             ELSE NULL
        END AS female_sales_ratio,

        CASE WHEN total_sales > 0
             THEN (age_20_sales + age_30_sales)::NUMERIC / total_sales
             ELSE NULL
        END AS young_adult_sales_ratio,

        -- QoQ change (LAG window)
        LAG(total_sales) OVER (
            PARTITION BY trdar_cd, svc_induty_cd
            ORDER BY stdr_yyqu_cd
        ) AS prev_q_sales,

        LAG(total_txn_count) OVER (
            PARTITION BY trdar_cd, svc_induty_cd
            ORDER BY stdr_yyqu_cd
        ) AS prev_q_txn_count
    FROM base
)
SELECT
    stdr_yyqu_cd,
    trdar_cd,
    trdar_cd_nm,
    svc_induty_cd,
    svc_induty_cd_nm,
    total_sales,
    total_txn_count,
    avg_sales_per_txn,
    weekend_sales_ratio,
    female_sales_ratio,
    young_adult_sales_ratio,

    -- QoQ sales change rate
    CASE WHEN prev_q_sales > 0
         THEN (total_sales - prev_q_sales)::NUMERIC / prev_q_sales
         ELSE NULL
    END AS sales_qoq_change,

    -- QoQ txn count change rate
    CASE WHEN prev_q_txn_count > 0
         THEN (total_txn_count - prev_q_txn_count)::NUMERIC / prev_q_txn_count
         ELSE NULL
    END AS txn_count_qoq_change,

    -- 2-quarter moving average ratio
    CASE WHEN prev_q_sales > 0
         THEN total_sales::NUMERIC / ((total_sales + prev_q_sales) / 2.0)
         ELSE NULL
    END AS sales_vs_2q_ma

FROM with_derived;

-- Indexes
CREATE INDEX idx_sales_q_keys
    ON staging.sales_quarterly(stdr_yyqu_cd, trdar_cd, svc_induty_cd);
