"""
Stage 4 — Model Development + External Data Ablation Study.

Core experiment:
  Model A: Card sales features only (ABP internal data simulation)
  Model B: Card sales + ALL external features
  Model C1: Card + Foot Traffic only
  Model C2: Card + District Change only
  Model C3: Card + Facilities only
  Model C4: Card + Macro (CSI) only
  Model C5: Card + Residents only

Answers:
  Q1: Which external features are actually useful?
  Q2: Do risk drivers differ by industry?
  Q3: How much does external data improve prediction?

Outputs:
  - figures/model_*.png
  - docs/model_results.md

Usage: python run_model.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/model_results.md"
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
# Feature Group Definitions
# ================================================================

# Group A: Card sales features (simulates ABP internal data)
GROUP_A = [
    "total_sales",
    "total_txn_count",
    "avg_sales_per_txn",
    "weekend_sales_ratio",
    "female_sales_ratio",
    "young_adult_sales_ratio",
    "sales_qoq_change",
    "txn_count_qoq_change",
    "sales_vs_2q_ma",
]

# Store context (always included - available from card data)
GROUP_STORE = [
    "store_count",
    "franchise_ratio",
    "competition_density",
    "closure_rate",
    "store_count_qoq_change",
]

# External feature groups
EXT_TRAFFIC = [
    "total_foot_traffic",
    "traffic_female_ratio",
    "traffic_young_adult_ratio",
    "traffic_qoq_change",
]

EXT_CHANGE = [
    "change_index_numeric",
    "operating_months_avg",
    "closed_months_avg",
]

EXT_FACILITIES = [
    "subway_count",
    "bus_stop_count",
    "total_facility_count",
]

EXT_MACRO = [
    "csi_avg",
]

EXT_RESIDENTS = [
    "total_residents",
    "total_households",
    "resident_young_adult_ratio",
    "resident_elderly_ratio",
]

ALL_EXTERNAL = EXT_TRAFFIC + EXT_CHANGE + EXT_FACILITIES + EXT_MACRO + EXT_RESIDENTS

TARGET = "high_risk_flag"


# ================================================================
# Data Loading + Temporal Split
# ================================================================
def load_and_split():
    """Load data and split by time (last 4 quarters = test)."""
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    log(f"Total rows: {len(df):,}")

    # Sort quarters
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    log(f"Quarters: {len(quarters)} ({quarters[0]} ~ {quarters[-1]})")

    # Temporal split: last 4 quarters = test
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]

    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()

    log(f"Train: {len(train_df):,} rows ({train_quarters[0]}~{train_quarters[-1]}, {len(train_quarters)} quarters)")
    log(f"Test: {len(test_df):,} rows ({test_quarters[0]}~{test_quarters[-1]}, {len(test_quarters)} quarters)")
    log(f"Train target rate: {train_df[TARGET].mean():.4f}")
    log(f"Test target rate: {test_df[TARGET].mean():.4f}")

    return train_df, test_df


# ================================================================
# Model Training + Evaluation
# ================================================================
def train_and_evaluate(train_df, test_df, feature_cols, model_name):
    """Train XGBoost and return metrics."""
    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[available].copy()
    y_test = test_df[TARGET].copy()

    # XGBoost handles NaN internally
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return {
        "model_name": model_name,
        "features": len(available),
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "model": model,
        "feature_names": available,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_baseline(train_df, test_df, feature_cols, model_name):
    """Train Logistic Regression baseline."""
    available = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[available].fillna(0).copy()
    y_train = train_df[TARGET].copy()
    X_test = test_df[available].fillna(0).copy()
    y_test = test_df[TARGET].copy()

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    return {
        "model_name": model_name,
        "features": len(available),
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ================================================================
# Ablation Study
# ================================================================
def run_ablation(train_df, test_df):
    """Run full A/B/C ablation study."""
    section("1. Ablation Study: Model A vs B vs C")

    base_features = GROUP_A + GROUP_STORE

    experiments = [
        ("LR Baseline (A)", base_features, "baseline"),
        ("Model A (Card Only)", base_features, "xgb"),
        ("Model B (Card + All External)", base_features + ALL_EXTERNAL, "xgb"),
        ("Model C1 (Card + Traffic)", base_features + EXT_TRAFFIC, "xgb"),
        ("Model C2 (Card + Change Index)", base_features + EXT_CHANGE, "xgb"),
        ("Model C3 (Card + Facilities)", base_features + EXT_FACILITIES, "xgb"),
        ("Model C4 (Card + Macro/CSI)", base_features + EXT_MACRO, "xgb"),
        ("Model C5 (Card + Residents)", base_features + EXT_RESIDENTS, "xgb"),
    ]

    results = []
    model_b_result = None

    for name, features, model_type in experiments:
        log(f"Training: {name} ({len(features)} features)")
        if model_type == "baseline":
            r = train_baseline(train_df, test_df, features, name)
        else:
            r = train_and_evaluate(train_df, test_df, features, name)

        results.append(r)
        log(f"  AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  Precision={r['precision']:.4f}  Recall={r['recall']:.4f}")

        if name == "Model B (Card + All External)":
            model_b_result = r

    # Results table
    log("")
    log("### Ablation Results")
    log("")
    log("| Model | Features | AUROC | F1 | Precision | Recall |")
    log("|---|---|---|---|---|---|")
    for r in results:
        log(f"| {r['model_name']} | {r['features']} | {r['auroc']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |")

    # Q3 answer: improvement
    model_a = next(r for r in results if "Model A" in r["model_name"])
    model_b = next(r for r in results if "Model B" in r["model_name"])
    delta_auroc = model_b["auroc"] - model_a["auroc"]
    delta_f1 = model_b["f1"] - model_a["f1"]

    log("")
    log(f"### Q3 Answer: External Data Improvement")
    log(f"- AUROC: {model_a['auroc']:.4f} (Card Only) -> {model_b['auroc']:.4f} (Card + External) = **+{delta_auroc:.4f}**")
    log(f"- F1: {model_a['f1']:.4f} -> {model_b['f1']:.4f} = **+{delta_f1:.4f}**")

    # Q1 answer: which external data matters most
    log("")
    log("### Q1 Answer: Individual External Data Contribution")
    log("")
    c_models = [r for r in results if r["model_name"].startswith("Model C")]
    c_models_sorted = sorted(c_models, key=lambda x: x["auroc"], reverse=True)

    log("| External Data | AUROC | Delta vs Card Only |")
    log("|---|---|---|")
    for r in c_models_sorted:
        delta = r["auroc"] - model_a["auroc"]
        log(f"| {r['model_name']} | {r['auroc']:.4f} | {'+' if delta >= 0 else ''}{delta:.4f} |")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["model_name"].replace("Model ", "").replace("(Card + ", "\n(+").replace("(Card Only)", "\n(Card Only)").replace("LR Baseline (A)", "LR\nBaseline") for r in results]
    aurocs = [r["auroc"] for r in results]
    colors = ["lightgray"] + ["steelblue"] + ["forestgreen"] + ["coral"] * 5

    bars = ax.bar(range(len(results)), aurocs, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation Study: Card-Only vs External Data")
    ax.axhline(model_a["auroc"], color="steelblue", linestyle="--", alpha=0.5, label=f"Model A baseline: {model_a['auroc']:.4f}")
    ax.legend()

    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_01_ablation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return results, model_b_result


# ================================================================
# SHAP Analysis
# ================================================================
def run_shap_analysis(model_b_result):
    """SHAP analysis on Model B for feature importance decomposition."""
    section("2. SHAP Analysis (Model B)")

    model = model_b_result["model"]
    X_test = model_b_result["X_test"]
    feature_names = model_b_result["feature_names"]

    # Sample for SHAP (full dataset can be slow)
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    log("### Global SHAP Feature Importance")
    log("")
    log("| Rank | Feature | Mean |SHAP| | Group |")
    log("|---|---|---|---|")
    for i, (_, row) in enumerate(importance_df.iterrows()):
        feat = row["feature"]
        if feat in GROUP_A:
            group = "Card Sales"
        elif feat in GROUP_STORE:
            group = "Store Context"
        elif feat in ALL_EXTERNAL:
            group = "External"
        else:
            group = "Other"
        log(f"| {i+1} | {feat} | {row['mean_abs_shap']:.4f} | {group} |")

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title("SHAP Feature Importance (Model B)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_02_shap_summary.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    # Group-level SHAP contribution
    section("3. SHAP by Feature Group")

    group_shap = {}
    for feat, shap_val in zip(feature_names, mean_abs_shap):
        if feat in GROUP_A:
            group_shap.setdefault("Card Sales (A)", []).append(shap_val)
        elif feat in GROUP_STORE:
            group_shap.setdefault("Store Context", []).append(shap_val)
        elif feat in EXT_TRAFFIC:
            group_shap.setdefault("Foot Traffic", []).append(shap_val)
        elif feat in EXT_CHANGE:
            group_shap.setdefault("Change Index", []).append(shap_val)
        elif feat in EXT_FACILITIES:
            group_shap.setdefault("Facilities", []).append(shap_val)
        elif feat in EXT_MACRO:
            group_shap.setdefault("Macro (CSI)", []).append(shap_val)
        elif feat in EXT_RESIDENTS:
            group_shap.setdefault("Residents", []).append(shap_val)

    group_totals = {k: sum(v) for k, v in group_shap.items()}
    group_sorted = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)

    log("| Feature Group | Total SHAP Contribution |")
    log("|---|---|")
    for group, total in group_sorted:
        log(f"| {group} | {total:.4f} |")

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = [g[0] for g in group_sorted]
    values = [g[1] for g in group_sorted]
    colors = []
    for g in groups:
        if "Card" in g:
            colors.append("steelblue")
        elif "Store" in g:
            colors.append("lightsteelblue")
        else:
            colors.append("coral")
    ax.barh(range(len(groups)), values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ax.set_xlabel("Total Mean |SHAP|")
    ax.set_title("SHAP Contribution by Feature Group")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_03_shap_groups.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return importance_df


# ================================================================
# Industry-Level Analysis (Q2)
# ================================================================
def run_industry_analysis(train_df, test_df):
    """Compare feature importance across top industries."""
    section("4. Industry-Level Risk Drivers (Q2)")

    # Top 5 industries by row count
    top_industries = (
        train_df.groupby("svc_induty_cd_nm").size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    all_features = GROUP_A + GROUP_STORE + ALL_EXTERNAL
    industry_importances = {}

    for industry in top_industries:
        ind_train = train_df[train_df["svc_induty_cd_nm"] == industry]
        ind_test = test_df[test_df["svc_induty_cd_nm"] == industry]

        if len(ind_train) < 100 or ind_train[TARGET].sum() < 10:
            continue

        try:
            r = train_and_evaluate(ind_train, ind_test, all_features, industry)
            model = r["model"]
            importance = model.feature_importances_
            available = r["feature_names"]

            top5 = sorted(zip(available, importance), key=lambda x: x[1], reverse=True)[:5]
            industry_importances[industry] = {
                "auroc": r["auroc"],
                "top_features": top5,
            }
            log(f"\n**{industry}** (AUROC={r['auroc']:.4f}):")
            for feat, imp in top5:
                group = "External" if feat in ALL_EXTERNAL else "Card/Store"
                log(f"  - {feat}: {imp:.4f} [{group}]")
        except Exception as e:
            log(f"  [WARN] {industry}: {e}")

    return industry_importances


# ================================================================
# Report Generator
# ================================================================
def generate_report():
    """Write markdown report."""
    header = [
        "# Stage 4: Model Results + Ablation Study",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Core question: Does external data improve closure prediction",
        "beyond what card sales data alone can achieve?",
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
    print("Stage 4: Model Development + Ablation Study")
    print("=" * 60)

    train_df, test_df = load_and_split()

    results, model_b = run_ablation(train_df, test_df)

    if model_b and "model" in model_b:
        run_shap_analysis(model_b)

    run_industry_analysis(train_df, test_df)

    generate_report()
    print("\nStage 4 complete.")


if __name__ == "__main__":
    main()
