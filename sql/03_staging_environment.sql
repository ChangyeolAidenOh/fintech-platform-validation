-- ================================================================
-- Stage 2: staging.environment_quarterly
-- Sources: raw.foot_traffic_raw, raw.district_change_raw,
--          raw.facilities_raw, raw.residents_raw
-- Grain: stdr_yyqu_cd × trdar_cd
-- External data features (non-card-sales)
-- ================================================================

DROP TABLE IF EXISTS staging.environment_quarterly CASCADE;

CREATE TABLE staging.environment_quarterly AS
WITH traffic AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        COALESCE(tot_flpop_co, 0)             AS total_foot_traffic,
        COALESCE(ml_flpop_co, 0)              AS male_foot_traffic,
        COALESCE(fml_flpop_co, 0)             AS female_foot_traffic,
        COALESCE(agrde_20_flpop_co, 0)
            + COALESCE(agrde_30_flpop_co, 0)  AS young_adult_traffic,

        -- QoQ foot traffic change
        LAG(COALESCE(tot_flpop_co, 0)) OVER (
            PARTITION BY trdar_cd ORDER BY stdr_yyqu_cd
        ) AS prev_q_traffic
    FROM raw.foot_traffic_raw
),
traffic_derived AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        total_foot_traffic,

        -- Female ratio
        CASE WHEN total_foot_traffic > 0
             THEN female_foot_traffic::NUMERIC / total_foot_traffic
             ELSE NULL
        END AS traffic_female_ratio,

        -- Young adult ratio
        CASE WHEN total_foot_traffic > 0
             THEN young_adult_traffic::NUMERIC / total_foot_traffic
             ELSE NULL
        END AS traffic_young_adult_ratio,

        -- QoQ change
        CASE WHEN prev_q_traffic > 0
             THEN (total_foot_traffic - prev_q_traffic)::NUMERIC / prev_q_traffic
             ELSE NULL
        END AS traffic_qoq_change
    FROM traffic
),
change_ind AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        trdar_chg_ind       AS change_index_code,
        trdar_chg_ind_nm    AS change_index_name,
        -- Encode as numeric (higher = more risk)
        CASE trdar_chg_ind
            WHEN 'HH' THEN 3  -- 정체
            WHEN 'HL' THEN 2  -- 상권축소
            WHEN 'LH' THEN 1  -- 상권확장
            WHEN 'LL' THEN 0  -- 다이나믹
            ELSE NULL
        END AS change_index_numeric,
        COALESCE(opbiz_rt, 0)  AS operating_months_avg,
        COALESCE(clsbiz_rt, 0) AS closed_months_avg
    FROM raw.district_change_raw
),
facilities AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        COALESCE(subway_statn_co, 0)    AS subway_count,
        COALESCE(bus_sttn_co, 0)        AS bus_stop_count,
        COALESCE(bank_co, 0)            AS bank_count,
        COALESCE(gnrl_hsptl_co, 0)      AS hospital_count,
        COALESCE(pharmcy_co, 0)         AS pharmacy_count,
        COALESCE(kndrgrt_co, 0)
            + COALESCE(elesch_co, 0)
            + COALESCE(mskul_co, 0)
            + COALESCE(hgschl_co, 0)    AS school_count,
        COALESCE(univ_co, 0)            AS university_count,
        COALESCE(dprtm_str_co, 0)
            + COALESCE(supmk_co, 0)     AS retail_anchor_count,
        COALESCE(stayng_fclty_co, 0)    AS lodging_count,
        -- Total facility score
        COALESCE(subway_statn_co, 0)
            + COALESCE(bus_sttn_co, 0)
            + COALESCE(bank_co, 0)
            + COALESCE(gnrl_hsptl_co, 0)
            + COALESCE(pharmcy_co, 0)
            + COALESCE(kndrgrt_co, 0)
            + COALESCE(elesch_co, 0)
            + COALESCE(mskul_co, 0)
            + COALESCE(hgschl_co, 0)
            + COALESCE(univ_co, 0)
            + COALESCE(dprtm_str_co, 0)
            + COALESCE(supmk_co, 0)
            + COALESCE(theat_co, 0)
            + COALESCE(stayng_fclty_co, 0)
            + COALESCE(pblc_parkng_lot_co, 0)
            AS total_facility_count
    FROM raw.facilities_raw
),
residents AS (
    SELECT
        stdr_yyqu_cd,
        trdar_cd,
        COALESCE(tot_popltn_co, 0)      AS total_residents,
        COALESCE(tot_hshld_co, 0)       AS total_households,

        -- Young adult resident ratio
        CASE WHEN COALESCE(tot_popltn_co, 0) > 0
             THEN (COALESCE(agrde_20_popltn_co, 0)
                 + COALESCE(agrde_30_popltn_co, 0))::NUMERIC
                 / COALESCE(tot_popltn_co, 0)
             ELSE NULL
        END AS resident_young_adult_ratio,

        -- Elderly ratio
        CASE WHEN COALESCE(tot_popltn_co, 0) > 0
             THEN COALESCE(agrde_60_above_popltn_co, 0)::NUMERIC
                 / COALESCE(tot_popltn_co, 0)
             ELSE NULL
        END AS resident_elderly_ratio
    FROM raw.residents_raw
)
-- Full outer join on (quarter, district) to preserve all records
SELECT
    COALESCE(t.stdr_yyqu_cd, c.stdr_yyqu_cd, f.stdr_yyqu_cd, r.stdr_yyqu_cd)
        AS stdr_yyqu_cd,
    COALESCE(t.trdar_cd, c.trdar_cd, f.trdar_cd, r.trdar_cd)
        AS trdar_cd,

    -- Foot traffic features
    t.total_foot_traffic,
    t.traffic_female_ratio,
    t.traffic_young_adult_ratio,
    t.traffic_qoq_change,

    -- District change features
    c.change_index_code,
    c.change_index_name,
    c.change_index_numeric,
    c.operating_months_avg,
    c.closed_months_avg,

    -- Facility features
    f.subway_count,
    f.bus_stop_count,
    f.total_facility_count,

    -- Resident features
    r.total_residents,
    r.total_households,
    r.resident_young_adult_ratio,
    r.resident_elderly_ratio

FROM traffic_derived t
FULL OUTER JOIN change_ind c
    ON t.stdr_yyqu_cd = c.stdr_yyqu_cd AND t.trdar_cd = c.trdar_cd
FULL OUTER JOIN facilities f
    ON COALESCE(t.stdr_yyqu_cd, c.stdr_yyqu_cd) = f.stdr_yyqu_cd
    AND COALESCE(t.trdar_cd, c.trdar_cd) = f.trdar_cd
FULL OUTER JOIN residents r
    ON COALESCE(t.stdr_yyqu_cd, c.stdr_yyqu_cd, f.stdr_yyqu_cd) = r.stdr_yyqu_cd
    AND COALESCE(t.trdar_cd, c.trdar_cd, f.trdar_cd) = r.trdar_cd;

-- Indexes
CREATE INDEX idx_env_q_keys
    ON staging.environment_quarterly(stdr_yyqu_cd, trdar_cd);
