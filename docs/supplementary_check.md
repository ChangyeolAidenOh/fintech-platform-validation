# Supplementary Verification

**Generated:** 2026-05-11 03:50


## 1. Threshold Label Overlap Check

Issue: if most industry medians are 0, different multipliers produce identical labels.

Industries with median closure_rate = 0: 62/63 (98.4%)

### Label Overlap (Jaccard Similarity)

|  | 1.0x | 1.25x | 1.5x | 2.0x | 2.5x |
|---|---|---|---|---|---|
| 1.0x | 1.000 | 0.997 | 0.993 | 0.983 | 0.970 |
| 1.25x | 0.997 | 1.000 | 0.996 | 0.985 | 0.973 |
| 1.5x | 0.993 | 0.996 | 1.000 | 0.989 | 0.977 |
| 2.0x | 0.983 | 0.985 | 0.989 | 1.000 | 0.988 |
| 2.5x | 0.970 | 0.973 | 0.977 | 0.988 | 1.000 |

### Positive Rates

| Threshold | Positive Count | Positive Rate |
|---|---|---|
| 1.0x | 134,728 | 0.2363 |
| 1.25x | 134,369 | 0.2357 |
| 1.5x | 133,849 | 0.2348 |
| 2.0x | 132,374 | 0.2322 |
| 2.5x | 130,736 | 0.2293 |

Jaccard(1.0x, 2.5x) = 0.9704

**CONFIRMED: Labels are nearly identical across thresholds.**
Most industry medians are 0, so multiplier changes don't affect labels.
Threshold sensitivity test is effectively a repeated test on the same target.
Interpretation should note this as a structural limitation.

## 2. Pure External vs Broad External

operating_months_avg and closed_months_avg may carry quasi-target information.
Comparing: ALL_EXTERNAL vs PURE_EXTERNAL (without change-index vars) vs CHANGE_VARS only.

External broad (all 15 vars): AUROC=0.6216, AUPRC=0.3436
Pure environment (12 vars): AUROC=0.6202, AUPRC=0.3434
Change-index vars only (3 vars): AUROC=0.5523, AUPRC=0.2616

| Scenario | AUROC | AUPRC |
|---|---|---|
| External broad (all 15 vars) | 0.6216 | 0.3436 |
| Pure environment (12 vars) | 0.6202 | 0.3434 |
| Change-index vars only (3 vars) | 0.5523 | 0.2616 |

Broad vs Pure AUROC difference: +0.0014
Change-index vars do not materially inflate external-only performance.
Cold-start results are not driven by quasi-target leakage.

## 3. Early Stopping Fix: Internal Validation vs Test Set

Original: early stopping uses test set (mild optimistic bias risk).
Fixed: early stopping uses internal validation (last 4Q of train).
Compare to confirm ablation conclusions are unchanged.

| Method | Model | AUROC | AUPRC | Best Iter |
|---|---|---|---|---|
| Original (test ES) | Base | 0.7828 | 0.5771 | 285 |
| Original (test ES) | Full | 0.7807 | 0.5741 | 64 |
| Fixed (val ES) | Base | 0.7820 | 0.5757 | 224 |
| Fixed (val ES) | Full | 0.7803 | 0.5738 | 81 |

Original dAUROC (Base→Full): -0.0021
Fixed dAUROC (Base→Full): -0.0017
Difference: 0.0004

Ablation conclusion is unchanged regardless of early stopping method.
Test-set early stopping did not bias the relative comparison.
