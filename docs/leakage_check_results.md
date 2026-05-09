# Stage 4: Leakage Verification + Naive Baseline Check

**Generated:** 2026-05-05 23:05

Critical validation: Is the model actually learning, or just exploiting autocorrelation?

Train: 485,100 (20191~20243)
Test: 84,989 (20244~20253)
Train target rate: 0.2360
Test target rate: 0.2277
Time-based split: train ends at 20243, test starts at 20244
Same district-industry combos in both: 22098

## 1. Naive Baselines (How hard is this problem really?)

If naive baselines match XGBoost, the model learned nothing meaningful.

### Baseline 1: Previous Quarter Closure Rate
  AUROC: 0.6059
  Precision@100: 0.1900
  Precision@500: 0.1360

### Baseline 2: Industry Average Closure Rate
  AUROC: 0.6049
  Precision@100: 0.1200
  Precision@500: 0.1880

### Baseline 3: store_count Only
  AUROC: 0.7445
  Precision@100: 0.9900
  Precision@500: 0.9440

### Baseline 4: closure_rate Only (XGBoost, 1 feature)
  AUROC: 0.6387
  Precision@100: 0.8200
  Precision@500: 0.8040

## 2. Leakage Test: closure_rate Impact

closure_rate = current quarter's closure count / store count.
Target = NEXT quarter's closure rate (LEAD window).
Time separation exists, but autocorrelation may inflate performance.

Full (with closure_rate):
  AUROC=0.7833, F1=0.5341, P@100=1.0000, P@500=0.9860
Strict (without closure_rate):
  AUROC=0.7829, F1=0.5333, P@100=1.0000, P@500=0.9820
Full + External (with closure_rate):
  AUROC=0.7812, F1=0.5354, P@100=0.9900, P@500=0.9680
Strict + External (without closure_rate):
  AUROC=0.7809, F1=0.5354, P@100=0.9900, P@500=0.9740

### Leakage Impact Summary

| Model | Features | AUROC | F1 | P@100 | P@500 |
|---|---|---|---|---|---|
| Full (with closure_rate) | 14 | 0.7833 | 0.5341 | 1.0000 | 0.9860 |
| Strict (without closure_rate) | 13 | 0.7829 | 0.5333 | 1.0000 | 0.9820 |
| Full + External (with closure_rate) | 29 | 0.7812 | 0.5354 | 0.9900 | 0.9680 |
| Strict + External (without closure_rate) | 28 | 0.7809 | 0.5354 | 0.9900 | 0.9740 |

### closure_rate contribution: AUROC delta = 0.0004
closure_rate contribution is minimal. No significant leakage concern.

## 3. Strict Ablation (All Leaky Variables Removed)

This is the HONEST comparison: no closure_rate, no close_count,
no open_count, no net_store_change in features.
Only structural + behavioral features that are truly available BEFORE the prediction quarter.

Strict A (Card Only):     AUROC=0.7829, F1=0.5333, P@100=1.0000
Strict B (Card+External): AUROC=0.7809, F1=0.5354, P@100=0.9900
Delta AUROC: -0.0020
Delta F1: +0.0021

Null result holds even after removing leaky variables.
External public data genuinely adds no incremental value at this resolution.

## 4. Comprehensive Model Comparison


| Category | Model | AUROC | P@100 |
|---|---|---|---|
| Naive | Naive: prev closure_rate | 0.6059 | 0.1900 |
| Naive | Naive: industry avg | 0.6049 | 0.1200 |
| Naive | Naive: store_count only | 0.7445 | 0.9900 |
| Naive | XGB: closure_rate only | 0.6387 | 0.8200 |
| XGB (with leaky) | XGB: Full (with closure_rate) | 0.7833 | 1.0000 |
| XGB (with leaky) | XGB: Strict (without closure_rate) | 0.7829 | 1.0000 |
| XGB (with leaky) | XGB: Full + External (with closure_rate) | 0.7812 | 0.9900 |
| XGB (with leaky) | XGB: Strict + External (without closure_rate) | 0.7809 | 0.9900 |
| XGB (strict) | XGB Strict: Card Only | 0.7829 | 1.0000 |
| XGB (strict) | XGB Strict: Card+External | 0.7809 | 0.9900 |
- Figure: figures/model_10_comprehensive.png

### Key Takeaway
- Best naive baseline: Naive: store_count only (AUROC=0.7445)
- Best strict XGBoost: AUROC=0.7829
- XGBoost improvement over best naive: 0.0384

XGBoost provides moderate improvement over naive baselines.
