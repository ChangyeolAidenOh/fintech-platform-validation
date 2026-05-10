"""
Stage 4 Extended — Alternative Target: Percentile-Based High-Risk.

Current target: next_q_closure_rate > industry median × 1.5
Problem: 62/63 industries have median=0, so target ≈ "any closure at all"

Alternative target B: within each industry×quarter,
top 20% of closure rates = high-risk.
This avoids the median=0 problem and asks a genuinely different question:
"Among all areas in this industry, which ones close MORE than peers?"

Usage: python run_alt_target.py
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import xgboost as xgb
import shap
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/alt_target_results.md"
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
# Feature Groups
# ================================================================
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

EXT_TRAFFIC = [
    "total_foot_traffic", "traffic_female_ratio",
    "traffic_young_adult_ratio", "traffic_qoq_change",
]
EXT_CHANGE = ["change_index_numeric", "operating_months_avg", "closed_months_avg"]
EXT_FACILITIES = ["subway_count", "bus_stop_count", "total_facility_count"]
EXT_MACRO = ["csi_avg"]
EXT_RESIDENTS = [
    "total_residents", "total_households",
    "resident_young_adult_ratio", "resident_elderly_ratio",
]


# ================================================================
# Data Loading + Alternative Target Construction
# ================================================================
def load_and_build_targets():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    log(f"Loaded {len(df):,} rows")

    # Original target
    log(f"Original target (high_risk_flag) positive rate: {df['high_risk_flag'].mean():.4f}")

    # Alternative Target B: top 20% within industry×quarter
    df["pctl_rank"] = df.groupby(
        ["stdr_yyqu_cd", "svc_induty_cd"]
    )["next_q_closure_rate"].rank(pct=True, method="average")

    df["alt_target_top20"] = (df["pctl_rank"] > 0.80).astype(int)
    df["alt_target_top30"] = (df["pctl_rank"] > 0.70).astype(int)
    df["alt_target_top10"] = (df["pctl_rank"] > 0.90).astype(int)

    log(f"Alt target top20% positive rate: {df['alt_target_top20'].mean():.4f}")
    log(f"Alt target top30% positive rate: {df['alt_target_top30'].mean():.4f}")
    log(f"Alt target top10% positive rate: {df['alt_target_top10'].mean():.4f}")

    # Check label overlap with original
    overlap = ((df["high_risk_flag"] == 1) & (df["alt_target_top20"] == 1)).sum()
    union = ((df["high_risk_flag"] == 1) | (df["alt_target_top20"] == 1)).sum()
    jaccard = overlap / max(union, 1)
    log(f"\nJaccard(original, top20%): {jaccard:.4f}")
    log("(Low Jaccard = genuinely different target)")

    return df


def time_split(df):
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]
    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()
    return train_df, test_df


# ================================================================
# Model Training (with internal validation for early stopping)
# ================================================================
def train_xgb(train_df, test_df, feature_cols, target):
    available = [c for c in feature_cols if c in train_df.columns]
    if len(available) == 0:
        return None

    # Internal validation split
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
    y_pred = model.predict(X_test)

    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "model": model,
        "feature_names": available,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
    }


# ================================================================
# 1. Compare Original vs Alternative Target
# ================================================================
def compare_targets(df):
    section("1. Original vs Percentile-Based Target")

    train_df, test_df = time_split(df)
    base_features = SALES_FEATURES + STORE_FEATURES

    targets = [
        ("Original (median×1.5)", "high_risk_flag"),
        ("Percentile top 20%", "alt_target_top20"),
        ("Percentile top 30%", "alt_target_top30"),
        ("Percentile top 10%", "alt_target_top10"),
    ]

    results = []
    for name, target in targets:
        r = train_xgb(train_df, test_df, base_features, target)
        if r:
            results.append({"Target": name, "target_col": target, **r})
            log(f"{name}: AUROC={r['auroc']:.4f}, AUPRC={r['auprc']:.4f}, F1={r['f1']:.4f}")

    log("")
    log("| Target | Pos Rate | AUROC | AUPRC | F1 |")
    log("|---|---|---|---|---|")
    for r in results:
        pos_rate = test_df[r["target_col"]].mean()
        log(f"| {r['Target']} | {pos_rate:.1%} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['f1']:.4f} |")

    return results


# ================================================================
# 2. Ablation on Alternative Target (top 20%)
# ================================================================
def alt_target_ablation(df):
    section("2. Ablation on Percentile Top 20% Target")

    log("Does external data help when the target is 'relatively worse than peers'")
    log("instead of 'any closure at all'?")
    log("")

    train_df, test_df = time_split(df)
    target = "alt_target_top20"

    base_features = SALES_FEATURES + STORE_FEATURES

    experiments = [
        ("Sales+Store (A)", base_features),
        ("A + All External (B)", base_features + ALL_EXTERNAL),
        ("A + Traffic", base_features + EXT_TRAFFIC),
        ("A + Change Index", base_features + EXT_CHANGE),
        ("A + Facilities", base_features + EXT_FACILITIES),
        ("A + CSI", base_features + EXT_MACRO),
        ("A + Residents", base_features + EXT_RESIDENTS),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, target)
        if r:
            results.append({"Model": name, **r})

    baseline = results[0]

    log("| Model | AUROC | AUPRC | dAUROC | dAUPRC |")
    log("|---|---|---|---|---|")
    for r in results:
        da = r["auroc"] - baseline["auroc"]
        dp = r["auprc"] - baseline["auprc"]
        da_str = f"{da:+.4f}" if r != baseline else "—"
        dp_str = f"{dp:+.4f}" if r != baseline else "—"
        log(f"| {r['Model']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {da_str} | {dp_str} |")

    # Key comparison
    model_a = baseline
    model_b = next(r for r in results if "All External" in r["Model"])
    delta = model_b["auroc"] - model_a["auroc"]
    delta_p = model_b["auprc"] - model_a["auprc"]

    log(f"\n### Key Result (Percentile Target)")
    log(f"Sales+Store → +External: dAUROC={delta:+.4f}, dAUPRC={delta_p:+.4f}")

    if delta > 0.005:
        log("\n**NEW FINDING: With percentile target, external data shows meaningful lift.**")
        log("The median=0 target was masking external data's value.")
    elif delta > 0.001:
        log("\nMarginal signal from external data on percentile target.")
    else:
        log("\nNull result persists even with percentile-based target.")
        log("External data genuinely adds no incremental value regardless of target definition.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Ablation bars
    names = [r["Model"] for r in results]
    aurocs = [r["auroc"] for r in results]
    deltas = [r["auroc"] - baseline["auroc"] for r in results]

    colors = ["#4682B4" if i == 0 else ("#2E8B57" if "All" in n else "#E8744F") for i, n in enumerate(names)]
    axes[0].bar(range(len(results)), aurocs, color=colors, edgecolor="black", alpha=0.8)
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels([n.replace("(", "\n(").replace(" + ", "\n+") for n in names], fontsize=7)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Percentile Top 20% Target: Ablation")
    axes[0].set_ylim([0, max(aurocs) * 1.1])

    for i, v in enumerate(aurocs):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=7)

    # Right: Delta comparison original vs percentile
    comparison = pd.DataFrame({
        "Target": ["Original\n(median×1.5)", "Percentile\n(top 20%)"],
        "dAUROC": [
            -0.0021,  # from main ablation
            delta,
        ],
    })
    colors2 = ["#E8744F" if d < 0 else "#2E8B57" for d in comparison["dAUROC"]]
    axes[1].bar(range(2), comparison["dAUROC"], color=colors2, edgecolor="black", alpha=0.8)
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(comparison["Target"])
    axes[1].set_ylabel("ΔAUROC (A → B)")
    axes[1].set_title("External Data Lift: Original vs Percentile Target")
    axes[1].axhline(0, color="black", linewidth=0.5)

    for i, v in enumerate(comparison["dAUROC"]):
        axes[1].text(i, v + 0.0005 if v >= 0 else v - 0.001, f"{v:+.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_16_alt_target.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return results


# ================================================================
# 3. SHAP on Alternative Target (if lift found)
# ================================================================
def alt_target_shap(df):
    section("3. SHAP on Percentile Target (Model B)")

    train_df, test_df = time_split(df)
    features = SALES_FEATURES + STORE_FEATURES + ALL_EXTERNAL
    target = "alt_target_top20"

    r = train_xgb(train_df, test_df, features, target)
    if not r:
        log("Could not train model.")
        return

    model = r["model"]
    X_test = r["X_test"]
    feature_names = r["feature_names"]

    sample_size = min(3000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log("| Rank | Feature | Mean |SHAP| | Group |")
    log("|---|---|---|---|")
    for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
        feat = row["feature"]
        if feat in SALES_FEATURES:
            group = "Sales"
        elif feat in STORE_FEATURES:
            group = "Store"
        else:
            group = "External"
        log(f"| {i+1} | {feat} | {row['mean_abs_shap']:.4f} | {group} |")

    # Group-level
    group_shap = {"Sales": 0, "Store": 0, "External": 0}
    for feat, sv in zip(feature_names, mean_abs_shap):
        if feat in SALES_FEATURES:
            group_shap["Sales"] += sv
        elif feat in STORE_FEATURES:
            group_shap["Store"] += sv
        else:
            group_shap["External"] += sv

    total = sum(group_shap.values())
    log("\n### SHAP Group Contribution (Percentile Target)")
    log("")
    log("| Group | Total SHAP | Pct |")
    log("|---|---|---|")
    for g, v in sorted(group_shap.items(), key=lambda x: x[1], reverse=True):
        log(f"| {g} | {v:.4f} | {v/total:.1%} |")


# ================================================================
# 4. Cold-Start on Alternative Target
# ================================================================
def alt_cold_start(df):
    section("4. Cold-Start on Percentile Target")

    train_df, test_df = time_split(df)
    target = "alt_target_top20"

    scenarios = [
        ("Sales+Store+External", SALES_FEATURES + STORE_FEATURES + ALL_EXTERNAL),
        ("Sales+Store (no ext)", SALES_FEATURES + STORE_FEATURES),
        ("Store+External (no sales)", STORE_FEATURES + ALL_EXTERNAL),
        ("Store only", STORE_FEATURES),
        ("External only", ALL_EXTERNAL),
    ]

    results = []
    for name, features in scenarios:
        r = train_xgb(train_df, test_df, features, target)
        if r:
            results.append({"Scenario": name, **r})

    log("| Scenario | AUROC | AUPRC |")
    log("|---|---|---|")
    for r in results:
        log(f"| {r['Scenario']} | {r['auroc']:.4f} | {r['auprc']:.4f} |")

    # No-sales comparison
    store_only = next((r for r in results if r["Scenario"] == "Store only"), None)
    store_ext = next((r for r in results if r["Scenario"] == "Store+External (no sales)"), None)
    if store_only and store_ext:
        delta = store_ext["auroc"] - store_only["auroc"]
        log(f"\nNo-sales: Store→Store+External lift: {delta:+.4f}")


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Stage 4 Extended: Alternative Target (Percentile-Based)",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Does the null result hold when target avoids the median=0 problem?",
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
    print("Stage 4 Extended: Alternative Target (Percentile-Based)")
    print("=" * 60)

    df = load_and_build_targets()

    compare_targets(df)

    alt_target_ablation(df)

    alt_target_shap(df)

    alt_cold_start(df)

    generate_report()
    print("\nDone.")


if __name__ == "__main__":
    main()
