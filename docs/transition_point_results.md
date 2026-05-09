# Stage 4 Branch 3: Transition Point Prediction

**Generated:** 2026-05-05 21:47

Predicting the first closure event in previously healthy trade areas.


## 1. Transition Point Dataset Construction

Full dataset: 570,089 rows
Rows with zero current closure: 438,933 (77.0%)
Transition events: 78,310 (17.8%)
Non-transitions: 360,623 (82.2%)
Train: 372,933 (transition rate: 0.1793)
Test: 66,000 (transition rate: 0.1731)

## 2. Transition Point Ablation (A vs B)

Does external data matter MORE for transition prediction
than for general closure rate prediction?

Model A (Card Only): AUROC=0.7372, F1=0.4233, Prec=0.3169, Rec=0.6372
Model B (Card + All External): AUROC=0.7352, F1=0.4230, Prec=0.3199, Rec=0.6241

Delta AUROC (A→B): -0.0020

**Finding:** External data does not provide more value for transition
prediction. The transition is driven by the same factors as general closure.
- Figure: figures/model_08_transition_ablation.png

## 3. SHAP for Transition Prediction (Model B)

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | store_count | 0.5027 | Store |
| 2 | franchise_ratio | 0.1826 | Store |
| 3 | store_count_qoq_change | 0.1325 | Store |
| 4 | competition_density | 0.0811 | Store |
| 5 | total_sales | 0.0694 | Card |
| 6 | csi_avg | 0.0627 | External |
| 7 | weekend_sales_ratio | 0.0621 | Card |
| 8 | avg_sales_per_txn | 0.0600 | Card |
| 9 | female_sales_ratio | 0.0506 | Card |
| 10 | txn_count_qoq_change | 0.0409 | Card |
| 11 | young_adult_sales_ratio | 0.0312 | Card |
| 12 | total_txn_count | 0.0150 | Card |
| 13 | total_residents | 0.0124 | External |
| 14 | operating_months_avg | 0.0083 | External |
| 15 | total_foot_traffic | 0.0077 | External |

External features in top 10: 1/10
- Figure: figures/model_09_transition_shap.png
