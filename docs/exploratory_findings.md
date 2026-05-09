# Stage 3: Exploratory Findings

**Generated:** 2026-05-05 19:48
**Source:** mart.risk_features

Loaded 570,089 rows from mart.risk_features
Columns: 41
Quarters: 27 (20191 ~ 20253)
Districts: 1603
Industries: 63

## 1. Target Variable Distribution

- Count: 570,089
- Mean: 0.0423
- Median: 0.0000
- Std: 0.1350
- Min: 0.0000, Max: 4.0000
- Zero closure rate: 434,973 (76.3%)
- High risk (flag=1): 133,849 (23.5%)
- Figure: figures/eda_01_target_distribution.png

## 2. Industry-Level Closure Rate

Top 10 highest closure rate industries:

| Industry | Mean | Median | Std | Count |
|---|---|---|---|---|
| 고시원 | 0.1402 | 0.0000 | 0.4713 | 18 |
| 치킨전문점 | 0.1336 | 0.0000 | 0.3213 | 13,282 |
| 편의점 | 0.1329 | 0.0000 | 0.3329 | 15,242 |
| 패스트푸드점 | 0.1053 | 0.0000 | 0.2687 | 9,879 |
| PC방 | 0.0828 | 0.0000 | 0.2192 | 3,777 |
| 분식전문점 | 0.0669 | 0.0000 | 0.1541 | 20,818 |
| 제과점 | 0.0653 | 0.0000 | 0.1817 | 11,047 |
| 커피-음료 | 0.0626 | 0.0000 | 0.1349 | 27,418 |
| 중식음식점 | 0.0568 | 0.0000 | 0.1525 | 11,702 |
| 일식음식점 | 0.0551 | 0.0000 | 0.1363 | 8,958 |

Bottom 5 lowest closure rate industries:

| Industry | Mean | Median | Count |
|---|---|---|---|
| 가전제품 | 0.0155 | 0.0000 | 1,760 |
| 조명용품 | 0.0127 | 0.0000 | 4,537 |
| 컴퓨터및주변장치판매 | 0.0123 | 0.0000 | 2,968 |
| 치과의원 | 0.0114 | 0.0000 | 10,275 |
| 일반의원 | 0.0106 | 0.0000 | 13,669 |

Kruskal-Wallis test: H=36493.79, p=0.00e+00
Conclusion: Industries have significantly different closure rates
- Figure: figures/eda_02_industry_closure.png

## 3. Sales QoQ Change vs Closure Rate

- Valid pairs: 531,032
- Spearman correlation: r=-0.0003, p=7.99e-01
- Interpretation: Negative correlation (sales decline -> higher closure)
- Figure: figures/eda_03_sales_vs_closure.png

## 4. Foot Traffic QoQ Change vs Closure Rate (External #1)

- Valid pairs: 546,003
- Spearman correlation: r=0.0061, p=6.47e-06
- Figure: figures/eda_04_traffic_vs_closure.png

## 5. District Change Index vs Closure Rate (External #2)

| Change Index | Mean Closure | Median Closure | Count |
|---|---|---|---|
| 다이나믹 | 0.0462 | 0.0000 | 213,526 |
| 상권확장 | 0.0421 | 0.0000 | 73,697 |
| 상권축소 | 0.0415 | 0.0000 | 122,217 |
| 정체 | 0.0378 | 0.0000 | 160,649 |

Chi-square test (change_index vs high_risk): chi2=3399.42, p=0.00e+00
Conclusion: Significant association
- Figure: figures/eda_05_change_index_vs_closure.png

## 6. Feature Correlation with Target

Spearman correlations with next_q_closure_rate:

| Feature | Correlation |
|---|---|
| change_index_numeric | -0.0614 |
| avg_sales_per_txn | -0.0471 |
| female_sales_ratio | -0.0387 |
| resident_elderly_ratio | -0.0380 |
| csi_avg | -0.0227 |
| sales_qoq_change | -0.0047 |
| txn_count_qoq_change | -0.0044 |
| traffic_qoq_change | 0.0061 |
| store_count_qoq_change | 0.0483 |
| resident_young_adult_ratio | 0.0602 |
| weekend_sales_ratio | 0.0783 |
| subway_count | 0.0998 |
| total_foot_traffic | 0.1036 *** |
| total_residents | 0.1096 *** |
| young_adult_sales_ratio | 0.1167 *** |
| total_facility_count | 0.1195 *** |
| competition_density | 0.1487 *** |
| franchise_ratio | 0.1487 *** |
| total_sales | 0.1673 *** |
| total_txn_count | 0.1732 *** |
| store_count | 0.3019 *** |
- Figure: figures/eda_06_correlation_heatmap.png

## 7. Missing Data Summary

| Column | Missing | Pct |
|---|---|---|
| csi_avg | 245,428 | 43.1% |
| total_facility_count | 140,647 | 24.7% |
| subway_count | 140,647 | 24.7% |
| bus_stop_count | 140,647 | 24.7% |
| txn_count_qoq_change | 27,379 | 4.8% |
| sales_qoq_change | 27,379 | 4.8% |
| sales_vs_2q_ma | 27,379 | 4.8% |
| store_count_qoq_change | 19,784 | 3.5% |
| traffic_qoq_change | 19,100 | 3.4% |
| total_residents | 928 | 0.2% |
| total_households | 928 | 0.2% |
| resident_young_adult_ratio | 928 | 0.2% |
| resident_elderly_ratio | 928 | 0.2% |
| franchise_ratio | 285 | 0.0% |
| closure_rate | 285 | 0.0% |
| competition_density | 285 | 0.0% |
| traffic_young_adult_ratio | 52 | 0.0% |
| traffic_female_ratio | 52 | 0.0% |
| total_foot_traffic | 52 | 0.0% |

## 8. Temporal Trend

- Figure: figures/eda_08_temporal_trend.png
