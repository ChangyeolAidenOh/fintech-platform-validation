"""
Stage 4 — Normalized Target + AUPRC.

Key insight: store_count alone gives P@100=99%, AUROC=0.74.
The model is learning "big areas have more closures" (size effect).
The real question: "Among similar-sized areas, why does one close more?"

Approach:
  1. Normalize target: excess closure rate = actual - industry average
  2. This removes size effect and industry baseline
  3. Re-run A vs B ablation on normalized target
  4. Add AUPRC as primary metric alongside AUROC

If external data shows effect on normalized target, it means:
"External data helps explain DEVIATIONS from expected closure,
 even though it doesn't help predict absolute closure levels."

Usage: python run_normalized_target.py
"""

# stdlib
import os
import warnings
from datetime import datetime

# third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
)
import xgboost as xgb
import shap
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/normalized_target_results.md"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

REPORT_LINES = []


def log(msg=""):
    print(msg)
    REPORT_LINES.append(msg)


def section(title):
    log("")
    log(f"## {title}")
    log("")


# ================================================================
# Feature Groups (strict — no leaky vars)
# ================================================================
GROUP_A_CLEAN = [
    "total_sales", "total_txn_count", "avg_sales_per_txn",
    "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
    "sales_qoq_change", "txn_count_qoq_change", "sales_vs_2q_ma",
]

GROUP_STORE_STRICT = [
    "store_count", "franchise_ratio", "competition_density",
    "store_count_qoq_change",
]

# Also test without store_count to fully remove size effect
GROUP_STORE_NO_SIZE = [
    "franchise_ratio", "competition_density",
    "store_count_qoq_change",
]

ALL_EXTERNAL = [
    "total_foot_traffic", "traffic_female_ratio", "traffic_young_adult_ratio",
    "traffic_qoq_change", "change_index_numeric", "operating_months_avg",
    "closed_months_avg", "subway_count", "bus_stop_count",
    "total_facility_count", "csi_avg", "total_residents",
    "total_households", "resident_young_adult_ratio", "resident_elderly_ratio",
]


# ================================================================
# Data Loading + Normalized Target
# ================================================================
def load_and_normalize():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    log(f"Loaded {len(df):,} rows")

    # Compute industry-quarter average closure rate
    industry_q_avg = (
        df.groupby(["stdr_yyqu_cd", "svc_induty_cd"])["next_q_closure_rate"]
        .transform("mean")
    )

    # Excess closure rate = actual - industry average
    df["excess_closure_rate"] = df["next_q_closure_rate"] - industry_q_avg

    # Binary target: above-average closure for this industry-quarter
    df["excess_high_risk"] = (df["excess_closure_rate"] > 0).astype(int)

    section("1. Normalized Target Distribution")

    log(f"Original target (high_risk_flag) positive rate: {df['high_risk_flag'].mean():.4f}")
    log(f"Normalized target (excess_high_risk) positive rate: {df['excess_high_risk'].mean():.4f}")
    log(f"Excess closure rate mean: {df['excess_closure_rate'].mean():.6f}")
    log(f"Excess closure rate std: {df['excess_closure_rate'].std():.6f}")
    log(f"Excess closure rate median: {df['excess_closure_rate'].median():.6f}")

    # Temporal split
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]

    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()

    log(f"\nTrain: {len(train_df):,}, Test: {len(test_df):,}")
    log(f"Train excess_high_risk rate: {train_df['excess_high_risk'].mean():.4f}")
    log(f"Test excess_high_risk rate: {test_df['excess_high_risk'].mean():.4f}")

    return train_df, test_df


# ================================================================
# Model Training with full metrics
# ================================================================
def train_xgb(train_df, test_df, feature_cols, model_name, target="excess_high_risk"):
    available = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[available].copy()
    y_train = train_df[target].copy()
    X_test = test_df[available].copy()
    y_test = test_df[target].copy()

    pos_count = max(y_train.sum(), 1)
    neg_count = (y_train == 0).sum()

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_count / pos_count,
        eval_metric="auc", early_stopping_rounds=30,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Precision@K
    sorted_idx = np.argsort(-y_prob)
    pk100 = y_test.iloc[sorted_idx[:100]].mean() if len(y_test) >= 100 else 0
    pk500 = y_test.iloc[sorted_idx[:500]].mean() if len(y_test) >= 500 else 0

    return {
        "model_name": model_name,
        "features": len(available),
        "feature_list": available,
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pk100": pk100,
        "pk500": pk500,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
    }


# ================================================================
# 2. Original vs Normalized Target Comparison
# ================================================================
def compare_targets(train_df, test_df):
    section("2. Original vs Normalized Target")

    log("Original target: high_risk_flag (closure_rate > industry median × 1.5)")
    log("Normalized target: excess_high_risk (closure_rate > industry-quarter average)")
    log("Normalized removes industry baseline and size effect.")
    log("")

    features_strict = GROUP_A_CLEAN + GROUP_STORE_STRICT

    # Original target
    r_orig = train_xgb(train_df, test_df, features_strict, "Original Target (Card)", target="high_risk_flag")
    # Normalized target
    r_norm = train_xgb(train_df, test_df, features_strict, "Normalized Target (Card)", target="excess_high_risk")

    log("| Target | AUROC | AUPRC | F1 | P@100 | P@500 |")
    log("|---|---|---|---|---|---|")
    log(f"| Original (high_risk) | {r_orig['auroc']:.4f} | {r_orig['auprc']:.4f} | {r_orig['f1']:.4f} | {r_orig['pk100']:.4f} | {r_orig['pk500']:.4f} |")
    log(f"| Normalized (excess) | {r_norm['auroc']:.4f} | {r_norm['auprc']:.4f} | {r_norm['f1']:.4f} | {r_norm['pk100']:.4f} | {r_norm['pk500']:.4f} |")

    log(f"\nAUROC drop from normalization: {r_orig['auroc'] - r_norm['auroc']:.4f}")
    log("(Drop is expected — we removed the 'easy' industry/size signal.)")
    log("The question is whether external data helps MORE on the harder, normalized task.")

    return r_orig, r_norm


# ================================================================
# 3. Normalized Target Ablation: A vs B
# ================================================================
def normalized_ablation(train_df, test_df):
    section("3. Normalized Target Ablation")

    log("With industry baseline removed, does external data help explain")
    log("WHY some areas deviate from their industry average?")
    log("")

    experiments = [
        ("Card + Store", GROUP_A_CLEAN + GROUP_STORE_STRICT),
        ("Card + Store + External", GROUP_A_CLEAN + GROUP_STORE_STRICT + ALL_EXTERNAL),
        ("Card + Store(no size) ", GROUP_A_CLEAN + GROUP_STORE_NO_SIZE),
        ("Card + Store(no size) + External", GROUP_A_CLEAN + GROUP_STORE_NO_SIZE + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name, target="excess_high_risk")
        results.append(r)
        log(f"{name}:")
        log(f"  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  F1={r['f1']:.4f}  P@100={r['pk100']:.4f}")

    # Key comparisons
    log("")
    log("### Ablation Results (Normalized Target)")
    log("")
    log("| Model | Feat | AUROC | AUPRC | F1 | P@100 | P@500 |")
    log("|---|---|---|---|---|---|---|")
    for r in results:
        log(f"| {r['model_name']} | {r['features']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1']:.4f} | {r['pk100']:.4f} | {r['pk500']:.4f} |")

    # With store_count
    a_size = next(r for r in results if r["model_name"] == "Card + Store")
    b_size = next(r for r in results if r["model_name"] == "Card + Store + External")
    delta_size = b_size["auroc"] - a_size["auroc"]
    delta_auprc_size = b_size["auprc"] - a_size["auprc"]

    log(f"\n### With store_count:")
    log(f"  AUROC delta (A→B): {'+' if delta_size >= 0 else ''}{delta_size:.4f}")
    log(f"  AUPRC delta (A→B): {'+' if delta_auprc_size >= 0 else ''}{delta_auprc_size:.4f}")

    # Without store_count
    a_nosize = next(r for r in results if r["model_name"] == "Card + Store(no size) ")
    b_nosize = next(r for r in results if r["model_name"] == "Card + Store(no size) + External")
    delta_nosize = b_nosize["auroc"] - a_nosize["auroc"]
    delta_auprc_nosize = b_nosize["auprc"] - a_nosize["auprc"]

    log(f"\n### Without store_count (full size-effect removal):")
    log(f"  AUROC delta (A→B): {'+' if delta_nosize >= 0 else ''}{delta_nosize:.4f}")
    log(f"  AUPRC delta (A→B): {'+' if delta_auprc_nosize >= 0 else ''}{delta_auprc_nosize:.4f}")

    # Interpretation
    log("")
    if delta_nosize > 0.005 or delta_auprc_nosize > 0.005:
        log("**NEW FINDING: With size effect AND industry baseline removed,**")
        log("**external data provides measurable value for explaining deviation from expected closure.**")
        log("Previous null result was driven by size/industry dominance masking external signal.")
    elif delta_nosize > 0.001 or delta_auprc_nosize > 0.001:
        log("Marginal signal from external data on normalized target.")
        log("External data may have weak but real explanatory power for closure deviations.")
    else:
        log("Null result persists even on normalized target.")
        log("External public data genuinely adds no value, regardless of target formulation.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: AUROC comparison
    model_names = [r["model_name"] for r in results]
    aurocs = [r["auroc"] for r in results]
    auprcs = [r["auprc"] for r in results]

    x = np.arange(len(results))
    w = 0.35
    axes[0].bar(x - w/2, aurocs, w, label="AUROC", alpha=0.8, edgecolor="black", color="steelblue")
    axes[0].bar(x + w/2, auprcs, w, label="AUPRC", alpha=0.8, edgecolor="black", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(" + ", "\n+ ") for n in model_names], fontsize=7)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Normalized Target: AUROC vs AUPRC")
    axes[0].legend()
    for i, (a, p) in enumerate(zip(aurocs, auprcs)):
        axes[0].text(i - w/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        axes[0].text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=7)

    # Right: Original vs Normalized comparison
    labels = ["Original\n(AUROC)", "Normalized\n(AUROC)", "Original\n(AUPRC)", "Normalized\n(AUPRC)"]
    # Use Card + Store results
    orig_r = train_xgb(train_df, test_df, GROUP_A_CLEAN + GROUP_STORE_STRICT,
                       "temp", target="high_risk_flag")
    norm_r = a_size
    vals = [orig_r["auroc"], norm_r["auroc"], orig_r["auprc"], norm_r["auprc"]]
    colors = ["steelblue", "lightsteelblue", "coral", "lightsalmon"]
    axes[1].bar(range(4), vals, color=colors, edgecolor="black", alpha=0.8)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Original vs Normalized Target")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_11_normalized_target.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return results


# ================================================================
# 4. SHAP on Normalized Target (best model)
# ================================================================
def normalized_shap(train_df, test_df):
    section("4. SHAP on Normalized Target (No Size Effect)")

    features = GROUP_A_CLEAN + GROUP_STORE_NO_SIZE + ALL_EXTERNAL
    r = train_xgb(train_df, test_df, features, "Normalized + No Size", target="excess_high_risk")

    model = r["model"]
    X_test = r["X_test"]
    feature_names = r["feature_list"]

    sample_size = min(3000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log("With industry baseline AND store_count removed,")
    log("which features explain why some areas deviate from expected closure?")
    log("")
    log("| Rank | Feature | Mean |SHAP| | Group |")
    log("|---|---|---|---|")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        feat = row["feature"]
        if feat in GROUP_A_CLEAN:
            group = "Card"
        elif feat in GROUP_STORE_NO_SIZE:
            group = "Store"
        else:
            group = "External"
        log(f"| {i+1} | {feat} | {row['mean_abs_shap']:.4f} | {group} |")

    # Count external in top 5 and top 10
    top5_ext = sum(1 for _, row in importance_df.head(5).iterrows() if row["feature"] in ALL_EXTERNAL)
    top10_ext = sum(1 for _, row in importance_df.head(10).iterrows() if row["feature"] in ALL_EXTERNAL)
    log(f"\nExternal features in top 5: {top5_ext}/5")
    log(f"External features in top 10: {top10_ext}/10")

    # Group-level contribution
    group_shap = {"Card": 0, "Store": 0, "External": 0}
    for feat, sv in zip(feature_names, mean_abs_shap):
        if feat in GROUP_A_CLEAN:
            group_shap["Card"] += sv
        elif feat in GROUP_STORE_NO_SIZE:
            group_shap["Store"] += sv
        else:
            group_shap["External"] += sv

    total_shap = sum(group_shap.values())
    log("\n### SHAP Group Contribution (Normalized, No Size)")
    log("")
    log("| Group | Total SHAP | Pct |")
    log("|---|---|---|")
    for g, v in sorted(group_shap.items(), key=lambda x: x[1], reverse=True):
        log(f"| {g} | {v:.4f} | {v/total_shap:.1%} |")

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP: Normalized Target (No Size Effect)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_12_normalized_shap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return importance_df


# ================================================================
# 5. AUPRC for All Previous Key Models (Retroactive)
# ================================================================
def auprc_retroactive(train_df, test_df):
    section("5. AUPRC for Key Models (Original Target)")

    log("AUPRC is more informative than AUROC for imbalanced classification.")
    log("Random baseline AUPRC = positive class rate.")
    log("")

    experiments = [
        ("Card + Store (strict)", GROUP_A_CLEAN + GROUP_STORE_STRICT, "high_risk_flag"),
        ("Card + Store + External (strict)", GROUP_A_CLEAN + GROUP_STORE_STRICT + ALL_EXTERNAL, "high_risk_flag"),
        ("Card + Store(no size)", GROUP_A_CLEAN + GROUP_STORE_NO_SIZE, "high_risk_flag"),
        ("Card + Store(no size) + External", GROUP_A_CLEAN + GROUP_STORE_NO_SIZE + ALL_EXTERNAL, "high_risk_flag"),
    ]

    log(f"Random baseline AUPRC: {test_df['high_risk_flag'].mean():.4f}")
    log("")
    log("| Model | AUROC | AUPRC | AUPRC Lift vs Random |")
    log("|---|---|---|---|")

    baseline_auprc = test_df["high_risk_flag"].mean()
    for name, features, target in experiments:
        r = train_xgb(train_df, test_df, features, name, target=target)
        lift = r["auprc"] / baseline_auprc
        log(f"| {name} | {r['auroc']:.4f} | {r['auprc']:.4f} | {lift:.2f}x |")


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Stage 4: Normalized Target + AUPRC Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Removing size effect and industry baseline to find hidden signals.",
        "",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        for line in header + REPORT_LINES:
            f.write(line + "\n")
    print(f"\nReport saved: {REPORT_PATH}")


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 60)
    print("Stage 4: Normalized Target + AUPRC Analysis")
    print("=" * 60)

    train_df, test_df = load_and_normalize()

    compare_targets(train_df, test_df)

    normalized_ablation(train_df, test_df)

    normalized_shap(train_df, test_df)

    auprc_retroactive(train_df, test_df)

    generate_report()
    print("\nNormalized target analysis complete.")


if __name__ == "__main__":
    main()
