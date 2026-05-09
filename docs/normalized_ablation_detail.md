# Stage 4: Normalized Target — Individual External Group Ablation

**Generated:** 2026-05-05 23:40

Resolving the SHAP 30.5% vs AUROC +0.001 paradox.

Train: 485,100, Test: 84,989
Target rate: 0.1900

## 1. Individual External Group Ablation (Normalized Target, With store_count)

| Model | AUROC | AUPRC | dAUROC | dAUPRC |
|---|---|---|---|---|
| A: Card + Store (baseline) | 0.7183 | 0.3586 | — | — |
| + Traffic | 0.7191 | 0.3617 | +0.0007 | +0.0032 |
| + Change Index | 0.7194 | 0.3647 | +0.0011 | +0.0061 |
| + Facilities | 0.7194 | 0.3620 | +0.0011 | +0.0034 |
| + CSI (Macro) | 0.7151 | 0.3510 | -0.0033 | -0.0075 |
| + Residents | 0.7200 | 0.3637 | +0.0017 | +0.0051 |
| + ALL External | 0.7172 | 0.3593 | -0.0012 | +0.0008 |

## 2. Individual External Group Ablation (Normalized Target, No store_count)

Full size-effect removal. This is where external data should shine if it has value.

| Model | AUROC | AUPRC | dAUROC | dAUPRC |
|---|---|---|---|---|
| A: Card + Store/noSize (baseline) | 0.6787 | 0.3285 | — | — |
| + Traffic | 0.6820 | 0.3311 | +0.0033 | +0.0026 |
| + Change Index | 0.6798 | 0.3326 | +0.0011 | +0.0040 |
| + Facilities | 0.6824 | 0.3315 | +0.0037 | +0.0029 |
| + CSI (Macro) | 0.6741 | 0.3239 | -0.0046 | -0.0046 |
| + Residents | 0.6827 | 0.3322 | +0.0040 | +0.0037 |
| + ALL External | 0.6799 | 0.3328 | +0.0012 | +0.0043 |

### Best individual external group: + Residents
  AUROC lift: +0.0040
  AUPRC lift: +0.0037

## 3. Interpretation: SHAP 30.5% vs AUROC +0.001

CSI alone (no store_count, normalized target):
  AUROC lift: -0.0046
  AUPRC lift: -0.0046

**Hypothesis B confirmed: SHAP contribution ≠ predictive improvement.**
External features redistribute SHAP credit but don't add new information.
They capture the same underlying signal as card/store features from a different angle.

### Why SHAP 30.5% but AUROC +0.001?

SHAP measures how much the model USES a feature for predictions.
AUROC measures whether ADDING that feature improves DISCRIMINATIVE ability.
These are different questions:
- If external features are correlated with card features,
  the model will USE them (SHAP > 0) but they won't ADD new info (AUROC ~ 0).
- This is feature redundancy, not uselessness.
- In practice, this means: if card data is unavailable, external data
  could serve as a partial substitute — but it doesn't enhance card data.
- Figure: figures/model_13_normalized_individual.png
