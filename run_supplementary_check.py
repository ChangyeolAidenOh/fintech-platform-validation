"""
Stage 4 Extended — Supplementary Verification.

1. Threshold label overlap check (median=0 → same labels?)
2. Pure external vs broad external (operating/closed_months_avg leakage risk)
3. Early stopping fix: use internal validation instead of test set

Usage: python run_supplementary_check.py
"""

# stdlib
import os
import warnings
from datetime import datetime

# third-party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

REPORT_PATH = "docs/supplementary_check.md"
os.makedirs("docs", exist_ok=True)

REPORT_LINES = []


def log(msg=""):
    print(msg)
    REPORT_LINES.append(msg)


def section(title):
    log("")
    log(f"## {title}")
    log("")


# Feature groups
SALES_FEATURES = [
    "total_sales", "total_txn_count", "avg_sales_per_txn",
    "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
    "sales_qoq_change", "txn_count_qoq_change", "sales_vs_2q_ma",
]

STORE_FEATURES = [
    "store_count", "franchise_ratio", "competition_density",
    "store_count_qoq_change",
]

ALL_EXTERNAL = [
    "total_foot_traffic", "traffic_female_ratio", "traffic_young_adult_ratio",
    "traffic_qoq_change", "change_index_numeric", "operating_months_avg",
    "closed_months_avg", "subway_count", "bus_stop_count",
    "total_facility_count", "csi_avg", "total_residents",
    "total_households", "resident_young_adult_ratio", "resident_elderly_ratio",
]

# Pure environment (no change-index derived vars)
PURE_EXTERNAL = [
    "total_foot_traffic", "traffic_female_ratio", "traffic_young_adult_ratio",
    "traffic_qoq_change",
    "subway_count", "bus_stop_count", "total_facility_count",
    "csi_avg", "total_residents", "total_households",
    "resident_young_adult_ratio", "resident_elderly_ratio",
]

# Change-index only
CHANGE_VARS = [
    "change_index_numeric", "operating_months_avg", "closed_months_avg",
]

TARGET = "high_risk_flag"


def load_data():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    return df


def time_split(df):
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_df = df[~df["stdr_yyqu_cd"].isin(test_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()
    return train_df, test_df


def train_xgb_strict(train_df, test_df, feature_cols, target=TARGET):
    """Train with internal validation split instead of test set for early stopping."""
    available = [c for c in feature_cols if c in train_df.columns]
    if len(available) == 0:
        return None

    # Internal validation: last 4 quarters of training period
    train_quarters = sorted(train_df["stdr_yyqu_cd"].unique())
    val_quarters = train_quarters[-4:]
    pure_train = train_df[~train_df["stdr_yyqu_cd"].isin(val_quarters)]
    val_df = train_df[train_df["stdr_yyqu_cd"].isin(val_quarters)]

    X_train = pure_train[available].copy()
    y_train = pure_train[target].copy()
    X_val = val_df[available].copy()
    y_val = val_df[target].copy()
    X_test = test_df[available].copy()
    y_test = test_df[target].copy()

    if y_train.sum() < 10 or y_test.sum() < 10:
        return None

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc", early_stopping_rounds=30,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "best_iteration": model.best_iteration,
    }


# ================================================================
# 1. Threshold Label Overlap Check
# ================================================================
def threshold_label_overlap(df):
    section("1. Threshold Label Overlap Check")

    log("Issue: if most industry medians are 0, different multipliers produce identical labels.")
    log("")

    # Check how many industries have median=0
    industry_medians = df.groupby("svc_induty_cd")["next_q_closure_rate"].median()
    zero_median_count = (industry_medians == 0).sum()
    total_industries = len(industry_medians)
    log(f"Industries with median closure_rate = 0: {zero_median_count}/{total_industries} ({zero_median_count/total_industries:.1%})")
    log("")

    # Compute labels for each threshold
    multipliers = [1.0, 1.25, 1.5, 2.0, 2.5]
    labels = {}
    for mult in multipliers:
        industry_median = df.groupby("svc_induty_cd")["next_q_closure_rate"].transform("median")
        labels[mult] = (df["next_q_closure_rate"] > industry_median * mult).astype(int)

    # Compute label overlap (Jaccard similarity)
    log("### Label Overlap (Jaccard Similarity)")
    log("")
    header = "| " + " | ".join([""] + [f"{m}x" for m in multipliers]) + " |"
    log(header)
    log("|" + "---|" * (len(multipliers) + 1))

    for m1 in multipliers:
        row = f"| {m1}x |"
        for m2 in multipliers:
            intersection = ((labels[m1] == 1) & (labels[m2] == 1)).sum()
            union = ((labels[m1] == 1) | (labels[m2] == 1)).sum()
            jaccard = intersection / max(union, 1)
            row += f" {jaccard:.3f} |"
        log(row)

    # Positive rates
    log("")
    log("### Positive Rates")
    log("")
    log("| Threshold | Positive Count | Positive Rate |")
    log("|---|---|---|")
    for mult in multipliers:
        pos = labels[mult].sum()
        rate = labels[mult].mean()
        log(f"| {mult}x | {pos:,} | {rate:.4f} |")

    # Check if 1.0x and 2.5x are nearly identical
    jaccard_extreme = ((labels[1.0] == 1) & (labels[2.5] == 1)).sum() / max(((labels[1.0] == 1) | (labels[2.5] == 1)).sum(), 1)
    log(f"\nJaccard(1.0x, 2.5x) = {jaccard_extreme:.4f}")

    if jaccard_extreme > 0.95:
        log("\n**CONFIRMED: Labels are nearly identical across thresholds.**")
        log("Most industry medians are 0, so multiplier changes don't affect labels.")
        log("Threshold sensitivity test is effectively a repeated test on the same target.")
        log("Interpretation should note this as a structural limitation.")
    elif jaccard_extreme > 0.80:
        log("\nHigh overlap — threshold sensitivity is partially redundant.")
    else:
        log("\nReasonable label variation — threshold sensitivity is informative.")

    return labels


# ================================================================
# 2. Pure External vs Broad External
# ================================================================
def pure_external_check(df):
    section("2. Pure External vs Broad External")

    log("operating_months_avg and closed_months_avg may carry quasi-target information.")
    log("Comparing: ALL_EXTERNAL vs PURE_EXTERNAL (without change-index vars) vs CHANGE_VARS only.")
    log("")

    train_df, test_df = time_split(df)

    scenarios = [
        ("External broad (all 15 vars)", ALL_EXTERNAL),
        ("Pure environment (12 vars)", PURE_EXTERNAL),
        ("Change-index vars only (3 vars)", CHANGE_VARS),
    ]

    results = []
    for name, features in scenarios:
        r = train_xgb_strict(train_df, test_df, features)
        if r:
            results.append({"Scenario": name, **r})
            log(f"{name}: AUROC={r['auroc']:.4f}, AUPRC={r['auprc']:.4f}")

    log("")
    log("| Scenario | AUROC | AUPRC |")
    log("|---|---|---|")
    for r in results:
        log(f"| {r['Scenario']} | {r['auroc']:.4f} | {r['auprc']:.4f} |")

    # Interpretation
    if len(results) >= 2:
        broad = next(r for r in results if "broad" in r["Scenario"])
        pure = next(r for r in results if "Pure" in r["Scenario"])
        delta = broad["auroc"] - pure["auroc"]
        log(f"\nBroad vs Pure AUROC difference: {delta:+.4f}")

        if delta > 0.01:
            log("**WARNING: change-index vars contribute meaningfully.**")
            log("operating_months_avg / closed_months_avg may carry quasi-target signal.")
            log("Cold-start AUROC 0.624 may be inflated by these variables.")
        else:
            log("Change-index vars do not materially inflate external-only performance.")
            log("Cold-start results are not driven by quasi-target leakage.")


# ================================================================
# 3. Early Stopping Fix Validation
# ================================================================
def early_stopping_comparison(df):
    section("3. Early Stopping Fix: Internal Validation vs Test Set")

    log("Original: early stopping uses test set (mild optimistic bias risk).")
    log("Fixed: early stopping uses internal validation (last 4Q of train).")
    log("Compare to confirm ablation conclusions are unchanged.")
    log("")

    train_df, test_df = time_split(df)

    base_features = SALES_FEATURES + STORE_FEATURES
    full_features = base_features + ALL_EXTERNAL

    # Strict (internal validation)
    r_base_strict = train_xgb_strict(train_df, test_df, base_features)
    r_full_strict = train_xgb_strict(train_df, test_df, full_features)

    # Original (test set for early stopping) for comparison
    def train_xgb_original(train_df, test_df, feature_cols):
        available = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[available].copy()
        y_train = train_df[TARGET].copy()
        X_test = test_df[available].copy()
        y_test = test_df[TARGET].copy()

        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="auc", early_stopping_rounds=30,
            random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            "auroc": roc_auc_score(y_test, y_prob),
            "auprc": average_precision_score(y_test, y_prob),
            "best_iteration": model.best_iteration,
        }

    r_base_orig = train_xgb_original(train_df, test_df, base_features)
    r_full_orig = train_xgb_original(train_df, test_df, full_features)

    log("| Method | Model | AUROC | AUPRC | Best Iter |")
    log("|---|---|---|---|---|")
    log(f"| Original (test ES) | Base | {r_base_orig['auroc']:.4f} | {r_base_orig['auprc']:.4f} | {r_base_orig['best_iteration']} |")
    log(f"| Original (test ES) | Full | {r_full_orig['auroc']:.4f} | {r_full_orig['auprc']:.4f} | {r_full_orig['best_iteration']} |")
    log(f"| Fixed (val ES) | Base | {r_base_strict['auroc']:.4f} | {r_base_strict['auprc']:.4f} | {r_base_strict['best_iteration']} |")
    log(f"| Fixed (val ES) | Full | {r_full_strict['auroc']:.4f} | {r_full_strict['auprc']:.4f} | {r_full_strict['best_iteration']} |")

    delta_orig = r_full_orig["auroc"] - r_base_orig["auroc"]
    delta_strict = r_full_strict["auroc"] - r_base_strict["auroc"]

    log(f"\nOriginal dAUROC (Base→Full): {delta_orig:+.4f}")
    log(f"Fixed dAUROC (Base→Full): {delta_strict:+.4f}")
    log(f"Difference: {abs(delta_orig - delta_strict):.4f}")

    if abs(delta_orig - delta_strict) < 0.003:
        log("\nAblation conclusion is unchanged regardless of early stopping method.")
        log("Test-set early stopping did not bias the relative comparison.")
    else:
        log("\n**NOTE: Early stopping method affects relative comparison.**")
        log("Fixed method should be used for final reported numbers.")


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Supplementary Verification",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        for line in header + REPORT_LINES:
            f.write(line + "\n")
    print(f"\nReport saved: {REPORT_PATH}")


def main():
    print("=" * 60)
    print("Supplementary Verification")
    print("=" * 60)

    df = load_data()

    threshold_label_overlap(df)
    pure_external_check(df)
    early_stopping_comparison(df)

    generate_report()
    print("\nDone.")


if __name__ == "__main__":
    main()
