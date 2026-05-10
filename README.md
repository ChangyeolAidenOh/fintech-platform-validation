# Fintech Platform Validation

**Validating a fintech data platform's self-acknowledged limitations using public datasets — feature ablation, SHAP analysis, and business proposals**

> **Key Question:** 소상공인·금융기관 고객이 상권 리스크를 판단할 때, 공공 외부데이터는 추정매출·점포 구조 변수 대비 실제로 추가 예측가치를 제공하는가?
>
> **Finding:** 본 공공데이터 환경에서는 증분효과가 제한적이었으며, 외부데이터는 보강재보다 결제 이력이 부족한 경우의 약한 fallback signal로 해석됩니다.

[Live Dashboard](https://fintech-platform-validation.streamlit.app/) · [분석 보고서](docs/insights_and_proposals.md)

---

## Project Overview

국내 카드사의 금융 빅데이터 플랫폼을 직접 사용하며, 플랫폼이 안내하는 "외부데이터 결합 필요성"을 출발점으로, 서울시 공공데이터 290만 행을 활용해 공공 외부데이터의 실제 증분가치를 15개 검증 실험으로 분석했습니다.

**Independent Project** · 2026.05 · Changyeol (Aiden) Oh

| 항목 | 내용 |
|---|---|
| 데이터 | 서울 열린데이터광장 7개 소스 + 한국은행 ECOS |
| 규모 | 2,906,472행 (raw) → 570,089행 (mart) |
| 기간 | 2019Q1 ~ 2025Q3, 27개 분기 |
| 상권 | 1,603개 상권 × 63개 업종 |
| 모델 | XGBoost + Logistic Regression + Naive Baselines 4종 |
| 실험 | 15개 독립 검증 (ablation, SHAP, cold-start, rolling, percentile target) |
| 대시보드 | Streamlit 7탭 (고객 리포트 프로토타입 포함) |

---

## Pipeline

```
Stage 0   ABP 플랫폼 직접 사용 → 한계 6개 확인 → 검증 질문 도출
             │
Stage 1   데이터 수집 (7개 API, 290만 행)
             │
Stage 2   SQL ETL: raw → staging → mart (PostgreSQL 3-tier)
             │
Stage 3   EDA + 가설 검정 (Kruskal-Wallis, Spearman, Chi-square)
             │
Stage 4   XGBoost Ablation Study (15개 실험)
             ├── Base: 순방향/역방향 ablation, SHAP, Precision@K
             ├── Interaction: SHAP interaction, divergence features
             ├── Control: store_count 통제, leakage 검증, naive baseline
             ├── Normalized: 정규화 타겟, AUPRC, SHAP group 재분해
             ├── Extended: cold-start, segment, rolling, threshold
             ├── Supplementary: label overlap, pure external, early stopping fix
             └── Alt Target: percentile top 20% (median=0 문제 보완)
             │
Stage 5   인사이트 도출 + 대고객 시사점
             │
Stage 6   Streamlit 대시보드 (7탭, 고객 리포트 포함)
```

---

## Key Results

### 1. 외부데이터 증분효과

| 타겟 | 추정매출+점포 (A) | A + 전체 외부 (B) | ΔAUROC |
|---|---|---|---|
| Original (median×1.5) | 0.7830 | 0.7811 | -0.0018 |
| Percentile top 20% | 0.6909 | 0.6896 | -0.0012 |

두 타겟 모두에서, 5개 rolling window 모두에서, early stopping 방식 변경 후에도 **증분효과 제한적**.

### 2. SHAP vs AUROC 괴리 (Feature Redundancy)

| 조건 | 점포 구조 | 추정매출 | 외부 |
|---|---|---|---|
| 원본 타겟 | 73% | 19% | 6% |
| 정규화 (no store_count) | 36.6% | 32.9% | **30.5%** |

SHAP 기여도 30.5%이지만 AUROC 개선 +0.001. 외부 피처는 추정매출 피처와 **중복 신호(redundancy)**.

### 3. Cold-Start: 약한 Fallback Signal

| 조건 (Percentile target) | AUROC | AUPRC |
|---|---|---|
| Store only | 0.6734 | 0.2737 |
| Store + External | 0.6787 (+0.0053) | 0.2923 (+0.0186) |
| External only | 0.5762 | 0.2201 |

매출 정보가 없는 no-sales 조건에서는 외부데이터가 AUROC과 AUPRC를 모두 개선.

### 4. Naive Baseline 대비 실질 개선

| Model | AUROC |
|---|---|
| Naive: store_count 단독 | 0.7445 |
| XGB: 추정매출+점포 (strict) | 0.7829 |
| **실질 개선** | **+0.038** |

---

## Dashboard

<table>
  <tr>
    <td><strong>Tab 1:</strong> 프로젝트 동기 + KPI + Target Users</td>
    <td><strong>Tab 2:</strong> Ablation Study + ΔAUROC + Naive Baseline</td>
  </tr>
  <tr>
    <td><strong>Tab 3:</strong> 업종별 리스크 + 3-Tier 모니터링</td>
    <td><strong>Tab 4:</strong> SHAP + Feature Redundancy + 외부데이터 역할 분류</td>
  </tr>
  <tr>
    <td><strong>Tab 5:</strong> 시계열 트렌드 (Sanity Check)</td>
    <td><strong>Tab 6:</strong> 시사점 + 프로젝트 한계</td>
  </tr>
  <tr>
    <td colspan="2"><strong>Tab 7:</strong> 고객 리포트 (상권/업종 선택 → 리스크 등급 + Reason Code + 권장 액션)</td>
  </tr>
</table>

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11, SQL (PostgreSQL 16) |
| Database | PostgreSQL (Docker), psycopg2 |
| ML | XGBoost, scikit-learn |
| Interpretation | SHAP |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Statistics | SciPy (Kruskal-Wallis, Spearman, Chi-square) |
| Data Collection | Seoul Open Data API, Bank of Korea ECOS API |

---

## Project Structure

```
├── collectors/           # 7개 API 수집기 + seoul_api.py
├── database/             # connection.py, schema_init.sql
├── sql/                  # 01~04 ETL (raw → staging → mart)
├── dashboard/            # Streamlit app.py + customer_report_tab.py
├── docs/                 # 분석 결과 문서 9개
├── figures/              # 시각화 16장
├── run_*.py              # Stage별 실행 스크립트 12개
├── docker-compose.yml
└── requirements.txt
```

---

## How to Reproduce

```bash
# 1. Clone & setup
git clone https://github.com/ChangyeolAidenOh/fintech-platform-validation.git
cd fintech-platform-validation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Environment variables
cp .env.example .env
# Add: SEOUL_API_KEY, ECOS_API_KEY, POSTGRES_* credentials

# 3. Database
docker compose up -d

# 4. Data collection → ETL → Analysis
python spike_api_check.py       # API endpoint verification
python run_collectors.py        # Stage 1: data collection
python run_sql_etl.py           # Stage 2: raw → staging → mart
python run_eda.py               # Stage 3: EDA
python run_model.py             # Stage 4: base ablation

# 5. Dashboard
python run_export.py            # Export CSV for dashboard
streamlit run dashboard/app.py  # Launch dashboard
```

---

## Limitations

- 상권×업종 단위 공공 집계 데이터 (개별 가맹점 단위 아님)
- 인허가 정보, 사업자 등록 상태, 임대료 데이터 미확보
- COVID-19 기간 포함 (2019~2025)
- 서울 지역 한정, 분기 단위 시간 해상도

---

## License

This project is for portfolio and educational purposes.
