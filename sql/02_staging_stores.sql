-- ================================================================
-- Stage 2: staging.stores_quarterly
-- Source: raw.stores_raw
-- Grain: stdr_yyqu_cd × trdar_cd × svc_induty_cd
-- TARGET VARIABLE: closure_rate, high_risk flag
-- ================================================================

DROP TABLE IF EXISTS staging.stores_quarterly CASCADE;

CREATE TABLE staging.stores_quarterly AS
WITH base AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        trdar_cd_nm,
        svc_induty_cd,
        svc_induty_cd_nm,
        COALESCE(stor_co, 0)                AS store_count,
        COALESCE(opbiz_stor_co, 0)          AS open_count,
        COALESCE(clsbiz_stor_co, 0)         AS close_count,
        COALESCE(similr_induty_stor_co, 0)  AS similar_store_count,
        COALESCE(frc_stor_co, 0)            AS franchise_count,
        COALESCE(opbiz_rt, 0)               AS open_rate_raw,
        COALESCE(clsbiz_rt, 0)              AS close_rate_raw
    FROM raw.stores_raw
),
with_derived AS (
    SELECT
        *,
        -- Computed closure rate (verified vs raw)
        CASE WHEN store_count > 0
             THEN close_count::NUMERIC / store_count
             ELSE NULL
        END AS closure_rate,

        -- Franchise ratio
        CASE WHEN store_count > 0
             THEN franchise_count::NUMERIC / store_count
             ELSE NULL
        END AS franchise_ratio,

        -- Net store change
        open_count - close_count AS net_store_change,

        -- Competition density (similar stores per store)
        CASE WHEN store_count > 0
             THEN similar_store_count::NUMERIC / store_count
             ELSE NULL
        END AS competition_density,

        -- Previous quarter store count for trend
        LAG(store_count) OVER (
            PARTITION BY trdar_cd, svc_induty_cd
            ORDER BY stdr_yyqu_cd
        ) AS prev_q_store_count,

        LAG(close_count) OVER (
            PARTITION BY trdar_cd, svc_induty_cd
            ORDER BY stdr_yyqu_cd
        ) AS prev_q_close_count
    FROM base
),
with_target AS (
    SELECT
        *,
        -- Store count QoQ change
        CASE WHEN prev_q_store_count > 0
             THEN (store_count - prev_q_store_count)::NUMERIC / prev_q_store_count
             ELSE NULL
        END AS store_count_qoq_change,

        -- Next quarter closure rate (LEAD) = TARGET for prediction
        LEAD(closure_rate) OVER (
            PARTITION BY trdar_cd, svc_induty_cd
            ORDER BY stdr_yyqu_cd
        ) AS next_q_closure_rate
    FROM with_derived
)
SELECT
    stdr_yyqu_cd,
    trdar_cd,
    trdar_cd_nm,
    svc_induty_cd,
    svc_induty_cd_nm,
    store_count,
    open_count,
    close_count,
    net_store_change,
    franchise_count,
    franchise_ratio,
    closure_rate,
    close_rate_raw,
    competition_density,
    store_count_qoq_change,
    next_q_closure_rate
FROM with_target;

-- Indexes
CREATE INDEX idx_stores_q_keys
    ON staging.stores_quarterly(stdr_yyqu_cd, trdar_cd, svc_induty_cd);
