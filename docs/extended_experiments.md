# Stage 4 Extended: Cold-Start, Segments, Rolling, Threshold

**Generated:** 2026-05-11 03:15

When does external data matter? Does the null result hold universally?

Loaded 570,089 rows

## 1. Cold-Start Simulation

When card/sales data is unavailable (new merchants, cash-heavy businesses),
does external data provide value as a substitute?

Full (Sales+Store+External): AUROC=0.7807, AUPRC=0.5741
Sales+Store (no external): AUROC=0.7828, AUPRC=0.5771
Store+External (no sales): AUROC=0.7742, AUPRC=0.5650
Store only (no sales, no external): AUROC=0.7730, AUPRC=0.5595
Limited sales+Store+External: AUROC=0.7775, AUPRC=0.5695
Limited sales+Store (no external): AUROC=0.7780, AUPRC=0.5691
External only (cold-start): AUROC=0.6239, AUPRC=0.3464

### Cold-Start Results

| Scenario | Features | AUROC | AUPRC |
|---|---|---|---|
| Full (Sales+Store+External) | 28 | 0.7807 | 0.5741 |
| Sales+Store (no external) | 13 | 0.7828 | 0.5771 |
| Store+External (no sales) | 19 | 0.7742 | 0.5650 |
| Store only (no sales, no external) | 4 | 0.7730 | 0.5595 |
| Limited sales+Store+External | 22 | 0.7775 | 0.5695 |
| Limited sales+Store (no external) | 7 | 0.7780 | 0.5691 |
| External only (cold-start) | 15 | 0.6239 | 0.3464 |

### Key Comparisons

**No-sales scenario (cold-start proxy):**
  Store only: 0.7730
  Store + External: 0.7742
  External lift: +0.0012
  -> Marginal value from external data in cold-start scenario.

**Limited-sales scenario:**
  Limited sales + Store: 0.7780
  Limited sales + Store + External: 0.7775
  External lift: -0.0006

**Pure cold-start (external only):**
  AUROC: 0.6239
  (Random baseline = 0.50)

- Figure: figures/model_14_cold_start.png

## 2. Segment-Specific Ablation

Does external data help more in specific segments?

### Segment Results

| Segment | N_test | Pos% | AUROC(base) | AUROC(+ext) | dAUROC | dAUPRC |
|---|---|---|---|---|---|---|
| All (baseline) | 84,989 | 22.8% | 0.7828 | 0.7807 | -0.0021 | -0.0030 |
| store_count bottom 20% | 20,602 | 9.3% | 0.6656 | 0.6645 | -0.0011 | -0.0026 |
| store_count top 20% | 18,259 | 51.8% | 0.7318 | 0.7311 | -0.0007 | -0.0009 |
| High-turnover industries | 12,965 | 30.0% | 0.7549 | 0.7527 | -0.0022 | -0.0033 |
| Stable industries | 4,882 | 10.4% | 0.7920 | 0.7965 | +0.0045 | -0.0016 |

Best segment for external data: Stable industries (dAUROC=+0.0045)
Worst segment: High-turnover industries (dAUROC=-0.0022)
- Figure: figures/model_15_segment_ablation.png

## 3. Rolling Time Validation

Confirm that the null result is not period-specific.

| Test Period | N_test | AUROC(base) | AUROC(+ext) | dAUROC |
|---|---|---|---|---|
| 20234~20243 | 86,476 | 0.7853 | 0.7843 | -0.0010 |
| 20241~20244 | 86,096 | 0.7851 | 0.7845 | -0.0006 |
| 20242~20251 | 85,625 | 0.7855 | 0.7849 | -0.0007 |
| 20243~20252 | 85,198 | 0.7860 | 0.7852 | -0.0008 |
| 20244~20253 | 84,989 | 0.7828 | 0.7807 | -0.0021 |

Average dAUROC across windows: -0.0011
External data lift is consistently negligible across all time windows.

## 4. Threshold Sensitivity

Does the null result change with different high-risk thresholds?

| Threshold | Pos Rate | AUROC(base) | AUROC(+ext) | dAUROC |
|---|---|---|---|---|
| 1.0x median | 22.9% | 0.7860 | 0.7839 | -0.0021 |
| 1.25x median | 22.9% | 0.7845 | 0.7825 | -0.0020 |
| 1.5x median | 22.8% | 0.7828 | 0.7807 | -0.0021 |
| 2.0x median | 22.5% | 0.7782 | 0.7759 | -0.0023 |
| 2.5x median | 22.2% | 0.7738 | 0.7715 | -0.0023 |

Null result is robust across all threshold levels.
