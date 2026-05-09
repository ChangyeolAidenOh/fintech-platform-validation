# Stage 4 Branch 1: SHAP Interaction + Divergence + Precision@K

**Generated:** 2026-05-05 21:31

Loaded 570,089 rows
Divergence features created: ['sales_traffic_divergence', 'sales_up_traffic_down', 'sales_down_traffic_up', 'txn_vs_traffic_ratio', 'sales_per_traffic']
Train: 485,100, Test: 84,989

## 2. Divergence Feature Ablation

**Hypothesis:** Divergence features (card × external crossing signals) capture
information that neither card data nor external data provides independently.

Model B (baseline): AUROC=0.7811, F1=0.5351
Model B + Divergence: AUROC=0.7812, F1=0.5357

Delta AUROC: +0.0001
Delta F1: +0.0006

Conclusion: Divergence features do not meaningfully improve prediction.
Card data and external data operate on independent axes for this problem.

### Precision@K Comparison

**Model B (No Divergence):**

## 1. Precision@K Analysis

| K | Precision@K | True Positives in Top-K | Base Rate |
|---|---|---|---|
| 50 | 0.9800 | 49/50 | 0.2277 |
| 100 | 0.9900 | 99/100 | 0.2277 |
| 200 | 0.9900 | 198/200 | 0.2277 |
| 500 | 0.9660 | 483/500 | 0.2277 |

Interpretation: Base rate = 0.2277. Precision@K > base rate means the model is concentrating risk correctly.

**Model B + Divergence:**

## 1. Precision@K Analysis

| K | Precision@K | True Positives in Top-K | Base Rate |
|---|---|---|---|
| 50 | 0.9800 | 49/50 | 0.2277 |
| 100 | 0.9900 | 99/100 | 0.2277 |
| 200 | 0.9900 | 198/200 | 0.2277 |
| 500 | 0.9800 | 490/500 | 0.2277 |

Interpretation: Base rate = 0.2277. Precision@K > base rate means the model is concentrating risk correctly.

- Figure: figures/model_04_divergence.png

## 3. SHAP Interaction Analysis

Computing SHAP interaction values on 2000 samples...
(This may take a few minutes)

### Top 15 Feature Interaction Pairs

| Rank | Feature 1 | Feature 2 | Mean |Interaction| |
|---|---|---|---|
| 1 | store_count (Store) | franchise_ratio (Store) | 0.0476 |
| 2 | total_sales (Card) | store_count (Store) | 0.0214 | *CROSS*
| 3 | closure_rate (Store) | store_count_qoq_change (Store) | 0.0167 |
| 4 | store_count (Store) | csi_avg (External) | 0.0118 | *CROSS*
| 5 | avg_sales_per_txn (Card) | store_count (Store) | 0.0116 | *CROSS*
| 6 | txn_count_qoq_change (Card) | store_count (Store) | 0.0116 | *CROSS*
| 7 | weekend_sales_ratio (Card) | franchise_ratio (Store) | 0.0111 | *CROSS*
| 8 | female_sales_ratio (Card) | store_count (Store) | 0.0106 | *CROSS*
| 9 | total_sales (Card) | weekend_sales_ratio (Card) | 0.0101 |
| 10 | avg_sales_per_txn (Card) | female_sales_ratio (Card) | 0.0100 |
| 11 | avg_sales_per_txn (Card) | franchise_ratio (Store) | 0.0092 | *CROSS*
| 12 | franchise_ratio (Store) | store_count_qoq_change (Store) | 0.0078 |
| 13 | total_sales (Card) | franchise_ratio (Store) | 0.0072 | *CROSS*
| 14 | franchise_ratio (Store) | closure_rate (Store) | 0.0072 |
| 15 | store_count (Store) | competition_density (Store) | 0.0071 |

Cross-group interactions in top 15: 1/15
- Figure: figures/model_05_shap_interaction.png

### Key Interaction: sales_qoq × traffic_qoq = 0.0000
This answers: 'When sales decline AND foot traffic decline coincide,
does the risk amplify beyond what each factor contributes alone?'
