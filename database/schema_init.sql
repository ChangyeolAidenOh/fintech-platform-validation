-- BC카드 ABP 외부데이터 검증 프로젝트
-- PostgreSQL 3-Tier Schema: raw → staging → mart

-- ================================================================
-- SCHEMAS
-- ================================================================
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;

-- ================================================================
-- RAW SCHEMA — collectors가 적재하는 원본 데이터
-- ================================================================

-- 추정매출 (상권 × 업종 × 분기)
CREATE TABLE IF NOT EXISTS raw.sales_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),      -- 기준년분기코드 (e.g. 20241)
    trdar_cd            VARCHAR(10),     -- 상권코드
    trdar_cd_nm         VARCHAR(100),    -- 상권코드명
    svc_induty_cd       VARCHAR(10),     -- 서비스업종코드
    svc_induty_cd_nm    VARCHAR(100),    -- 서비스업종코드명
    thsmon_selng_amt    NUMERIC(20,2),   -- 당월매출금액
    thsmon_selng_co     INTEGER,         -- 당월매출건수
    mdwk_selng_amt      NUMERIC(20,2),   -- 주중매출금액
    wkend_selng_amt     NUMERIC(20,2),   -- 주말매출금액
    ml_selng_amt        NUMERIC(20,2),   -- 남성매출금액
    fml_selng_amt       NUMERIC(20,2),   -- 여성매출금액
    agrde_10_selng_amt  NUMERIC(20,2),   -- 10대매출금액
    agrde_20_selng_amt  NUMERIC(20,2),   -- 20대매출금액
    agrde_30_selng_amt  NUMERIC(20,2),   -- 30대매출금액
    agrde_40_selng_amt  NUMERIC(20,2),   -- 40대매출금액
    agrde_50_selng_amt  NUMERIC(20,2),   -- 50대매출금액
    agrde_60_above_selng_amt NUMERIC(20,2), -- 60대이상매출금액
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- 점포 (상권 × 업종 × 분기)
CREATE TABLE IF NOT EXISTS raw.stores_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),
    trdar_cd            VARCHAR(10),
    trdar_cd_nm         VARCHAR(100),
    svc_induty_cd       VARCHAR(10),
    svc_induty_cd_nm    VARCHAR(100),
    stor_co             INTEGER,         -- 점포수
    similr_induty_stor_co INTEGER,       -- 유사업종점포수
    opbiz_rt            NUMERIC(10,4),   -- 개업률
    opbiz_stor_co       INTEGER,         -- 개업점포수
    clsbiz_rt           NUMERIC(10,4),   -- 폐업률
    clsbiz_stor_co      INTEGER,         -- 폐업점포수
    frc_stor_co         INTEGER,         -- 프랜차이즈점포수
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- 유동인구 (상권배후지 × 분기)
CREATE TABLE IF NOT EXISTS raw.foot_traffic_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),
    trdar_cd            VARCHAR(10),
    trdar_cd_nm         VARCHAR(100),
    tot_flpop_co        NUMERIC(15,2),   -- 총유동인구수
    ml_flpop_co         NUMERIC(15,2),   -- 남성유동인구수
    fml_flpop_co        NUMERIC(15,2),   -- 여성유동인구수
    agrde_10_flpop_co   NUMERIC(15,2),   -- 10대유동인구수
    agrde_20_flpop_co   NUMERIC(15,2),   -- 20대유동인구수
    agrde_30_flpop_co   NUMERIC(15,2),   -- 30대유동인구수
    agrde_40_flpop_co   NUMERIC(15,2),   -- 40대유동인구수
    agrde_50_flpop_co   NUMERIC(15,2),   -- 50대유동인구수
    agrde_60_above_flpop_co NUMERIC(15,2), -- 60대이상유동인구수
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- 상권변화지표 (상권 × 분기)
CREATE TABLE IF NOT EXISTS raw.district_change_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),
    trdar_cd            VARCHAR(10),
    trdar_cd_nm         VARCHAR(100),
    trdar_chg_ind       VARCHAR(20),     -- 상권변화지표
    trdar_chg_ind_nm    VARCHAR(50),     -- 상권변화지표명 (활성화/관심/침체 등)
    opbiz_rt            NUMERIC(10,4),   -- 개업률
    clsbiz_rt           NUMERIC(10,4),   -- 폐업률
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- 집객시설 (상권 × 분기)
CREATE TABLE IF NOT EXISTS raw.facilities_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),
    trdar_cd            VARCHAR(10),
    trdar_cd_nm         VARCHAR(100),
    bus_trminl_co       INTEGER,         -- 버스터미널수
    subway_statn_co     INTEGER,         -- 지하철역수
    bus_sttn_co         INTEGER,         -- 버스정류장수
    pblc_cmclt_co       INTEGER,         -- 관공서수
    bank_co             INTEGER,         -- 은행수
    gnrl_hsptl_co       INTEGER,         -- 종합병원수
    pharmcy_co          INTEGER,         -- 약국수
    kndrgrt_co          INTEGER,         -- 유치원수
    elesch_co           INTEGER,         -- 초등학교수
    mskul_co            INTEGER,         -- 중학교수
    hgschl_co           INTEGER,         -- 고등학교수
    univ_co             INTEGER,         -- 대학교수
    dprtm_str_co        INTEGER,         -- 백화점수
    supmk_co            INTEGER,         -- 슈퍼마켓수
    theat_co            INTEGER,         -- 극장수
    stayng_fclty_co     INTEGER,         -- 숙박시설수
    pblc_parkng_lot_co  INTEGER,         -- 공영주차장수
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- 상주인구 (상권배후지 × 분기)
CREATE TABLE IF NOT EXISTS raw.residents_raw (
    id                  SERIAL PRIMARY KEY,
    stdr_yyqu_cd        VARCHAR(6),
    trdar_cd            VARCHAR(10),
    trdar_cd_nm         VARCHAR(100),
    tot_popltn_co       INTEGER,         -- 총인구수
    ml_popltn_co        INTEGER,         -- 남성인구수
    fml_popltn_co       INTEGER,         -- 여성인구수
    agrde_10_popltn_co  INTEGER,         -- 10대인구수
    agrde_20_popltn_co  INTEGER,         -- 20대인구수
    agrde_30_popltn_co  INTEGER,         -- 30대인구수
    agrde_40_popltn_co  INTEGER,         -- 40대인구수
    agrde_50_popltn_co  INTEGER,         -- 50대인구수
    agrde_60_above_popltn_co INTEGER,    -- 60대이상인구수
    tot_hshld_co        INTEGER,         -- 총가구수
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- ECOS 거시지표 (월간)
CREATE TABLE IF NOT EXISTS raw.ecos_raw (
    id                  SERIAL PRIMARY KEY,
    stat_code           VARCHAR(20),     -- 통계표코드
    stat_name           VARCHAR(100),    -- 통계명
    item_code           VARCHAR(20),     -- 항목코드
    item_name           VARCHAR(100),    -- 항목명
    time_code           VARCHAR(10),     -- 시간코드 (YYYYMM)
    value               NUMERIC(20,4),   -- 값
    collected_at        TIMESTAMP DEFAULT NOW()
);

-- ================================================================
-- INDEXES (raw schema)
-- ================================================================
CREATE INDEX IF NOT EXISTS idx_sales_raw_keys ON raw.sales_raw(stdr_yyqu_cd, trdar_cd, svc_induty_cd);
CREATE INDEX IF NOT EXISTS idx_stores_raw_keys ON raw.stores_raw(stdr_yyqu_cd, trdar_cd, svc_induty_cd);
CREATE INDEX IF NOT EXISTS idx_traffic_raw_keys ON raw.foot_traffic_raw(stdr_yyqu_cd, trdar_cd);
CREATE INDEX IF NOT EXISTS idx_change_raw_keys ON raw.district_change_raw(stdr_yyqu_cd, trdar_cd);
CREATE INDEX IF NOT EXISTS idx_facilities_raw_keys ON raw.facilities_raw(stdr_yyqu_cd, trdar_cd);
CREATE INDEX IF NOT EXISTS idx_residents_raw_keys ON raw.residents_raw(stdr_yyqu_cd, trdar_cd);
CREATE INDEX IF NOT EXISTS idx_ecos_raw_keys ON raw.ecos_raw(stat_code, item_code, time_code);
