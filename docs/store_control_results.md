# Stage 4 Branch 2: Store Count Control + Reverse Ablation

**Generated:** 2026-05-05 21:45

Train: 485,100, Test: 84,989

## 1. Store Count Control Experiment

store_count dominates SHAP (63% of total contribution).
Removing it reveals whether external data has hidden predictive power
that was previously masked by store_count's dominance.

A-full (Card + Store): AUROC=0.7830, F1=0.5340
A-ctrl (Card + Store, no store_count): AUROC=0.7483, F1=0.5129
B-full (Card + Store + External): AUROC=0.7811, F1=0.5351
B-ctrl (Card + Store + External, no store_count): AUROC=0.7461, F1=0.5101

### Key Comparison (store_count removed)
- Card Only (no store_count): AUROC = 0.7483
- Card + External (no store_count): AUROC = 0.7461
- Delta: -0.0022

**Finding:** Even without store_count, external data does not
meaningfully improve prediction. The null result is robust.

## 1.1 SHAP without store_count (Model B-ctrl)

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | closure_rate | 0.3157 | Store |
| 2 | store_count_qoq_change | 0.2472 | Store |
| 3 | franchise_ratio | 0.2088 | Store |
| 4 | csi_avg | 0.0947 | External |
| 5 | weekend_sales_ratio | 0.0895 | Card |
| 6 | total_sales | 0.0774 | Card |
| 7 | total_txn_count | 0.0733 | Card |
| 8 | young_adult_sales_ratio | 0.0581 | Card |
| 9 | total_foot_traffic | 0.0516 | External |
| 10 | competition_density | 0.0510 | Store |
| 11 | female_sales_ratio | 0.0422 | Card |
| 12 | txn_count_qoq_change | 0.0408 | Card |
| 13 | total_residents | 0.0373 | External |
| 14 | change_index_numeric | 0.0364 | External |
| 15 | avg_sales_per_txn | 0.0313 | Card |
- Figure: figures/model_06_shap_no_storecount.png
- Figure: figures/model_07_store_control.png

## 2. Reverse Ablation (Remove one external group)

Forward ablation (add one group) showed minimal effect.
Reverse ablation (remove one group from full model) checks
if any group's REMOVAL causes a significant drop.

Model B (Full): AUROC=0.7811

| Removed Group | AUROC | Delta vs Full |
|---|---|---|
| - Traffic | 0.7814 | +0.0003 |
| - Change Index | 0.7814 | +0.0003 |
| - Facilities | 0.7813 | +0.0001 |
| - Macro (CSI) | 0.7843 | +0.0031 |
| - Residents | 0.7816 | +0.0005 |
| - ALL External | 0.7830 | +0.0018 |

If removing a group INCREASES AUROC, that group was adding noise.
If removing a group DECREASES AUROC, that group was contributing signal.
