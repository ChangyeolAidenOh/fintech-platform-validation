# Stage 4 Extended: Alternative Target (Percentile-Based)

**Generated:** 2026-05-11 04:05

Does the null result hold when target avoids the median=0 problem?

Loaded 570,089 rows
Original target (high_risk_flag) positive rate: 0.2348
Alt target top20% positive rate: 0.1704
Alt target top30% positive rate: 0.2092
Alt target top10% positive rate: 0.0977

Jaccard(original, top20%): 0.7257
(Low Jaccard = genuinely different target)

## 1. Original vs Percentile-Based Target

Original (median×1.5): AUROC=0.7820, AUPRC=0.5757, F1=0.5324
Percentile top 20%: AUROC=0.6909, AUPRC=0.3027, F1=0.3747
Percentile top 30%: AUROC=0.7471, AUPRC=0.4549, F1=0.4732
Percentile top 10%: AUROC=0.6363, AUPRC=0.1566, F1=0.2251

| Target | Pos Rate | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| Original (median×1.5) | 22.8% | 0.7820 | 0.5757 | 0.5324 |
| Percentile top 20% | 16.9% | 0.6909 | 0.3027 | 0.3747 |
| Percentile top 30% | 20.7% | 0.7471 | 0.4549 | 0.4732 |
| Percentile top 10% | 9.8% | 0.6363 | 0.1566 | 0.2251 |

## 2. Ablation on Percentile Top 20% Target

Does external data help when the target is 'relatively worse than peers'
instead of 'any closure at all'?

| Model | AUROC | AUPRC | dAUROC | dAUPRC |
|---|---|---|---|---|
| Sales+Store (A) | 0.6909 | 0.3027 | — | — |
| A + All External (B) | 0.6896 | 0.3033 | -0.0012 | +0.0006 |
| A + Traffic | 0.6909 | 0.3024 | +0.0000 | -0.0003 |
| A + Change Index | 0.6906 | 0.3021 | -0.0003 | -0.0006 |
| A + Facilities | 0.6919 | 0.3067 | +0.0010 | +0.0040 |
| A + CSI | 0.6870 | 0.2940 | -0.0039 | -0.0087 |
| A + Residents | 0.6913 | 0.3030 | +0.0004 | +0.0003 |

### Key Result (Percentile Target)
Sales+Store → +External: dAUROC=-0.0012, dAUPRC=+0.0006

Null result persists even with percentile-based target.
External data genuinely adds no incremental value regardless of target definition.
- Figure: figures/model_16_alt_target.png

## 3. SHAP on Percentile Target (Model B)

| Rank | Feature | Mean |SHAP| | Group |
|---|---|---|---|
| 1 | store_count | 0.4904 | Store |
| 2 | franchise_ratio | 0.1796 | Store |
| 3 | store_count_qoq_change | 0.1072 | Store |
| 4 | csi_avg | 0.1015 | External |
| 5 | txn_count_qoq_change | 0.0889 | Sales |
| 6 | total_sales | 0.0752 | Sales |
| 7 | avg_sales_per_txn | 0.0596 | Sales |
| 8 | weekend_sales_ratio | 0.0590 | Sales |
| 9 | female_sales_ratio | 0.0479 | Sales |
| 10 | competition_density | 0.0466 | Store |
| 11 | change_index_numeric | 0.0306 | External |
| 12 | total_txn_count | 0.0254 | Sales |
| 13 | young_adult_sales_ratio | 0.0225 | Sales |
| 14 | total_residents | 0.0220 | External |
| 15 | resident_elderly_ratio | 0.0189 | External |

### SHAP Group Contribution (Percentile Target)

| Group | Total SHAP | Pct |
|---|---|---|
| Store | 0.8239 | 55.1% |
| Sales | 0.3964 | 26.5% |
| External | 0.2740 | 18.3% |

## 4. Cold-Start on Percentile Target

| Scenario | AUROC | AUPRC |
|---|---|---|
| Sales+Store+External | 0.6896 | 0.3033 |
| Sales+Store (no ext) | 0.6909 | 0.3027 |
| Store+External (no sales) | 0.6787 | 0.2923 |
| Store only | 0.6734 | 0.2737 |
| External only | 0.5762 | 0.2201 |

No-sales: Store→Store+External lift: +0.0053
