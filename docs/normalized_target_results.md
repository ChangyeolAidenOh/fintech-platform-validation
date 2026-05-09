# Stage 4: Normalized Target + AUPRC Analysis

**Generated:** 2026-05-05 23:19

Removing size effect and industry baseline to find hidden signals.

Loaded 570,089 rows

## 1. Normalized Target Distribution

Original target (high_risk_flag) positive rate: 0.2348
Normalized target (excess_high_risk) positive rate: 0.1972
Excess closure rate mean: 0.000000
Excess closure rate std: 0.131689
Excess closure rate median: -0.025422

Train: 485,100, Test: 84,989
Train excess_high_risk rate: 0.1984
Test excess_high_risk rate: 0.1900

## 2. Original vs Normalized Target

Original target: high_risk_flag (closure_rate > industry median × 1.5)
Normalized target: excess_high_risk (closure_rate > industry-quarter average)
Normalized removes industry baseline and size effect.

| Target | AUROC | AUPRC | F1 | P@100 | P@500 |
|---|---|---|---|---|---|
| Original (high_risk) | 0.7829 | 0.5774 | 0.5333 | 1.0000 | 0.9820 |
| Normalized (excess) | 0.7183 | 0.3586 | 0.4300 | 0.6100 | 0.5180 |

AUROC drop from normalization: 0.0645
(Drop is expected — we removed the 'easy' industry/size signal.)
The question is whether external data helps MORE on the harder, normalized task.

## 3. Normalized Target Ablation

With industry baseline removed, does external data help explain
WHY some areas deviate from their industry average?

Card + Store:
  AUROC=0.7183  AUPRC=0.3586  F1=0.4300  P@100=0.6100
Card + Store + External:
  AUROC=0.7172  AUPRC=0.3593  F1=0.4297  P@100=0.6100
Card + Store(no size) :
  AUROC=0.6787  AUPRC=0.3285  F1=0.4031  P@100=0.4900
Card + Store(no size) + External:
  AUROC=0.6799  AUPRC=0.3328  F1=0.4011  P@100=0.4800

### Ablation Results (Normalized Target)

| Model | Feat | AUROC | AUPRC | F1 | P@100 | P@500 |
|---|---|---|---|---|---|---|
| Card + Store | 13 | 0.7183 | 0.3586 | 0.4300 | 0.6100 | 0.5180 |
| Card + Store + External | 28 | 0.7172 | 0.3593 | 0.4297 | 0.6100 | 0.5580 |
| Card + Store(no size)  | 12 | 0.6787 | 0.3285 | 0.4031 | 0.4900 | 0.5100 |
| Card + Store(no size) + External | 27 | 0.6799 | 0.3328 | 0.4011 | 0.4800 | 0.4940 |

### With store_count:
  AUROC delta (A→B): -0.0012
  AUPRC delta (A→B): +0.0008

### Without store_count (full size-effect removal):
  AUROC delta (A→B): +0.0012
  AUPRC delta (A→B): +0.0043

Marginal signal from external data on normalized target.
External data may have weak but real explanatory power for closure deviations.
- Figure: figures/model_11_normalized_target.png

## 4. SHAP on Normalized Target (No Size Effect)

With industry baseline AND store_count removed,
which features explain why some areas deviate from expected closure?

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | store_count_qoq_change | 0.2764 | Store |
| 2 | franchise_ratio | 0.2221 | Store |
| 3 | csi_avg | 0.1341 | External |
| 4 | weekend_sales_ratio | 0.0944 | Card |
| 5 | total_txn_count | 0.0898 | Card |
| 6 | txn_count_qoq_change | 0.0888 | Card |
| 7 | female_sales_ratio | 0.0632 | Card |
| 8 | competition_density | 0.0563 | Store |
| 9 | total_foot_traffic | 0.0560 | External |
| 10 | young_adult_sales_ratio | 0.0557 | Card |
| 11 | avg_sales_per_txn | 0.0538 | Card |
| 12 | total_residents | 0.0445 | External |
| 13 | change_index_numeric | 0.0413 | External |
| 14 | resident_elderly_ratio | 0.0353 | External |
| 15 | total_facility_count | 0.0306 | External |

External features in top 5: 1/5
External features in top 10: 2/10

### SHAP Group Contribution (Normalized, No Size)

| Group | Total SHAP | Pct |
|---|---|---|
| Store | 0.5548 | 36.6% |
| Card | 0.4981 | 32.9% |
| External | 0.4620 | 30.5% |
- Figure: figures/model_12_normalized_shap.png

## 5. AUPRC for Key Models (Original Target)

AUPRC is more informative than AUROC for imbalanced classification.
Random baseline AUPRC = positive class rate.

Random baseline AUPRC: 0.2277

| Model | AUROC | AUPRC | AUPRC Lift vs Random |
|---|---|---|---|
| Card + Store (strict) | 0.7829 | 0.5774 | 2.54x |
| Card + Store + External (strict) | 0.7809 | 0.5742 | 2.52x |
| Card + Store(no size) | 0.7423 | 0.5237 | 2.30x |
| Card + Store(no size) + External | 0.7404 | 0.5209 | 2.29x |
