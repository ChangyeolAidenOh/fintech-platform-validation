# Stage 4: Model Results + Ablation Study

**Generated:** 2026-05-05 20:14

Core question: Does external data improve closure prediction
beyond what card sales data alone can achieve?

Total rows: 570,089
Quarters: 27 (20191 ~ 20253)
Train: 485,100 rows (20191~20243, 23 quarters)
Test: 84,989 rows (20244~20253, 4 quarters)
Train target rate: 0.2360
Test target rate: 0.2277

## 1. Ablation Study: Model A vs B vs C

Training: LR Baseline (A) (14 features)
  AUROC=0.6325  F1=0.3990  Precision=0.2783  Recall=0.7041
Training: Model A (Card Only) (14 features)
  AUROC=0.7830  F1=0.5340  Precision=0.4453  Recall=0.6668
Training: Model B (Card + All External) (29 features)
  AUROC=0.7811  F1=0.5351  Precision=0.4560  Recall=0.6472
Training: Model C1 (Card + Traffic) (18 features)
  AUROC=0.7836  F1=0.5351  Precision=0.4465  Recall=0.6675
Training: Model C2 (Card + Change Index) (17 features)
  AUROC=0.7834  F1=0.5334  Precision=0.4455  Recall=0.6643
Training: Model C3 (Card + Facilities) (17 features)
  AUROC=0.7837  F1=0.5358  Precision=0.4495  Recall=0.6630
Training: Model C4 (Card + Macro/CSI) (15 features)
  AUROC=0.7813  F1=0.5345  Precision=0.4538  Recall=0.6502
Training: Model C5 (Card + Residents) (18 features)
  AUROC=0.7840  F1=0.5343  Precision=0.4470  Recall=0.6639

### Ablation Results

| Model | Features | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| LR Baseline (A) | 14 | 0.6325 | 0.3990 | 0.2783 | 0.7041 |
| Model A (Card Only) | 14 | 0.7830 | 0.5340 | 0.4453 | 0.6668 |
| Model B (Card + All External) | 29 | 0.7811 | 0.5351 | 0.4560 | 0.6472 |
| Model C1 (Card + Traffic) | 18 | 0.7836 | 0.5351 | 0.4465 | 0.6675 |
| Model C2 (Card + Change Index) | 17 | 0.7834 | 0.5334 | 0.4455 | 0.6643 |
| Model C3 (Card + Facilities) | 17 | 0.7837 | 0.5358 | 0.4495 | 0.6630 |
| Model C4 (Card + Macro/CSI) | 15 | 0.7813 | 0.5345 | 0.4538 | 0.6502 |
| Model C5 (Card + Residents) | 18 | 0.7840 | 0.5343 | 0.4470 | 0.6639 |

### Q3 Answer: External Data Improvement
- AUROC: 0.7830 (Card Only) -> 0.7811 (Card + External) = **+-0.0018**
- F1: 0.5340 -> 0.5351 = **+0.0011**

### Q1 Answer: Individual External Data Contribution

| External Data | AUROC | Delta vs Card Only |
|---|---|---|
| Model C5 (Card + Residents) | 0.7840 | +0.0010 |
| Model C3 (Card + Facilities) | 0.7837 | +0.0007 |
| Model C1 (Card + Traffic) | 0.7836 | +0.0007 |
| Model C2 (Card + Change Index) | 0.7834 | +0.0005 |
| Model C4 (Card + Macro/CSI) | 0.7813 | -0.0016 |
- Figure: figures/model_01_ablation.png

## 2. SHAP Analysis (Model B)

### Global SHAP Feature Importance

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | store_count | 0.6349 | Store Context |
| 2 | franchise_ratio | 0.1938 | Store Context |
| 3 | store_count_qoq_change | 0.1063 | Store Context |
| 4 | closure_rate | 0.1009 | Store Context |
| 5 | competition_density | 0.0810 | Store Context |
| 6 | csi_avg | 0.0591 | External |
| 7 | total_sales | 0.0574 | Card Sales |
| 8 | weekend_sales_ratio | 0.0550 | Card Sales |
| 9 | avg_sales_per_txn | 0.0538 | Card Sales |
| 10 | female_sales_ratio | 0.0493 | Card Sales |
| 11 | txn_count_qoq_change | 0.0308 | Card Sales |
| 12 | young_adult_sales_ratio | 0.0300 | Card Sales |
| 13 | total_txn_count | 0.0161 | Card Sales |
| 14 | total_residents | 0.0095 | External |
| 15 | total_foot_traffic | 0.0055 | External |
| 16 | operating_months_avg | 0.0055 | External |
| 17 | closed_months_avg | 0.0049 | External |
| 18 | change_index_numeric | 0.0048 | External |
| 19 | sales_vs_2q_ma | 0.0022 | Card Sales |
| 20 | total_facility_count | 0.0022 | External |
| 21 | resident_elderly_ratio | 0.0021 | External |
| 22 | sales_qoq_change | 0.0016 | Card Sales |
| 23 | resident_young_adult_ratio | 0.0013 | External |
| 24 | bus_stop_count | 0.0010 | External |
| 25 | traffic_female_ratio | 0.0009 | External |
| 26 | traffic_young_adult_ratio | 0.0008 | External |
| 27 | traffic_qoq_change | 0.0006 | External |
| 28 | subway_count | 0.0006 | External |
| 29 | total_households | 0.0000 | External |
- Figure: figures/model_02_shap_summary.png

## 3. SHAP by Feature Group

| Feature Group | Total SHAP Contribution |
|---|---|
| Store Context | 1.1169 |
| Card Sales (A) | 0.2963 |
| Macro (CSI) | 0.0591 |
| Change Index | 0.0152 |
| Residents | 0.0129 |
| Foot Traffic | 0.0078 |
| Facilities | 0.0037 |
- Figure: figures/model_03_shap_groups.png

## 4. Industry-Level Risk Drivers (Q2)


**한식음식점** (AUROC=0.7534):
  - store_count: 0.2963 [Card/Store]
  - total_txn_count: 0.1139 [Card/Store]
  - total_foot_traffic: 0.0527 [External]
  - total_sales: 0.0384 [Card/Store]
  - change_index_numeric: 0.0360 [External]

**미용실** (AUROC=0.7364):
  - store_count: 0.2468 [Card/Store]
  - total_txn_count: 0.0695 [Card/Store]
  - total_sales: 0.0595 [Card/Store]
  - store_count_qoq_change: 0.0510 [Card/Store]
  - competition_density: 0.0477 [Card/Store]

**커피-음료** (AUROC=0.7769):
  - store_count: 0.3095 [Card/Store]
  - total_sales: 0.1019 [Card/Store]
  - total_txn_count: 0.0816 [Card/Store]
  - total_foot_traffic: 0.0490 [External]
  - competition_density: 0.0381 [Card/Store]

**호프-간이주점** (AUROC=0.8007):
  - store_count: 0.2917 [Card/Store]
  - total_sales: 0.0892 [Card/Store]
  - closure_rate: 0.0622 [Card/Store]
  - bus_stop_count: 0.0349 [External]
  - store_count_qoq_change: 0.0346 [Card/Store]

**일반의류** (AUROC=0.7892):
  - store_count: 0.3347 [Card/Store]
  - closure_rate: 0.1023 [Card/Store]
  - total_sales: 0.0920 [Card/Store]
  - total_foot_traffic: 0.0284 [External]
  - total_facility_count: 0.0273 [External]
