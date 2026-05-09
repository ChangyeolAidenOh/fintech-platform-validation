"""
Stage 4 — Branch 2: Store Count Control + Reverse Ablation.

store_count dominates SHAP (63%). Removing it may reveal
hidden effects of external data that were previously masked.

Also runs reverse ablation: start from Model B, remove one
external group at a time, measure performance drop.

Usage: python run_store_control.py
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import shap
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/store_control_results.md"
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
GROUP_A = [
    "total_sales", "total_txn_count", "avg_sales_per_txn",
    "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
    "sales_qoq_change", "txn_count_qoq_change", "sales_vs_2q_ma",
]

# Store context WITHOUT store_count
GROUP_STORE_NO_COUNT = [
    "franchise_ratio", "competition_density",
    "closure_rate", "store_count_qoq_change",
]

GROUP_STORE_FULL = ["store_count"] + GROUP_STORE_NO_COUNT

EXT_TRAFFIC = [
    "total_foot_traffic", "traffic_female_ratio",
    "traffic_young_adult_ratio", "traffic_qoq_change",
]

EXT_CHANGE = [
    "change_index_numeric", "operating_months_avg", "closed_months_avg",
]

EXT_FACILITIES = [
    "subway_count", "bus_stop_count", "total_facility_count",
]

EXT_MACRO = ["csi_avg"]

EXT_RESIDENTS = [
    "total_residents", "total_households",
    "resident_young_adult_ratio", "resident_elderly_ratio",
]

ALL_EXTERNAL = EXT_TRAFFIC + EXT_CHANGE + EXT_FACILITIES + EXT_MACRO + EXT_RESIDENTS

TARGET = "high_risk_flag"


# ================================================================
# Data Loading
# ================================================================
def load_and_split():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]

    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()

    log(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    return train_df, test_df


# ================================================================
# Model Training
# ================================================================
def train_xgb(train_df, test_df, feature_cols, model_name):
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
    y_pred = model.predict(X_test)

    return {
        "model_name": model_name,
        "features": len(available),
        "auroc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "model": model,
        "feature_names": available,
        "X_test": X_test,
        "y_test": y_test,
    }


# ================================================================
# 1. Store Count Control Experiment
# ================================================================
def store_control_experiment(train_df, test_df):
    """Compare models with and without store_count."""
    section("1. Store Count Control Experiment")

    log("store_count dominates SHAP (63% of total contribution).")
    log("Removing it reveals whether external data has hidden predictive power")
    log("that was previously masked by store_count's dominance.")
    log("")

    experiments = [
        ("A-full (Card + Store)", GROUP_A + GROUP_STORE_FULL),
        ("A-ctrl (Card + Store, no store_count)", GROUP_A + GROUP_STORE_NO_COUNT),
        ("B-full (Card + Store + External)", GROUP_A + GROUP_STORE_FULL + ALL_EXTERNAL),
        ("B-ctrl (Card + Store + External, no store_count)", GROUP_A + GROUP_STORE_NO_COUNT + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name)
        results.append(r)
        log(f"{name}: AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}")

    # Key comparison: A-ctrl vs B-ctrl
    a_ctrl = next(r for r in results if "A-ctrl" in r["model_name"])
    b_ctrl = next(r for r in results if "B-ctrl" in r["model_name"])
    delta = b_ctrl["auroc"] - a_ctrl["auroc"]

    log(f"\n### Key Comparison (store_count removed)")
    log(f"- Card Only (no store_count): AUROC = {a_ctrl['auroc']:.4f}")
    log(f"- Card + External (no store_count): AUROC = {b_ctrl['auroc']:.4f}")
    log(f"- Delta: {'+' if delta >= 0 else ''}{delta:.4f}")

    if delta > 0.005:
        log(f"\n**Finding:** With store_count removed, external data provides")
        log(f"a meaningful AUROC improvement of {delta:.4f}.")
        log(f"store_count was masking the contribution of external features.")
    elif delta > 0.001:
        log(f"\n**Finding:** Marginal improvement ({delta:.4f}).")
        log(f"External data has some hidden value when store_count is removed.")
    else:
        log(f"\n**Finding:** Even without store_count, external data does not")
        log(f"meaningfully improve prediction. The null result is robust.")

    # SHAP on controlled model (B-ctrl)
    section("1.1 SHAP without store_count (Model B-ctrl)")

    model = b_ctrl["model"]
    X_test = b_ctrl["X_test"]
    feature_names = b_ctrl["feature_names"]

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
        if feat in GROUP_A:
            group = "Card"
        elif feat in GROUP_STORE_NO_COUNT:
            group = "Store"
        else:
            group = "External"
        log(f"| {i+1} | {feat} | {row['mean_abs_shap']:.4f} | {group} |")

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP Importance (store_count removed)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_06_shap_no_storecount.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    # Visualization: 4-model comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [r["model_name"].split("(")[0].strip() for r in results]
    aurocs = [r["auroc"] for r in results]
    colors = ["steelblue", "lightsteelblue", "forestgreen", "lightgreen"]

    bars = ax.bar(range(len(results)), aurocs, color=colors, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Store Count Control: Full vs Controlled Models")

    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_07_store_control.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    return results


# ================================================================
# 2. Reverse Ablation (Model B - remove one group at a time)
# ================================================================
def reverse_ablation(train_df, test_df):
    """Start from Model B, remove one external group at a time."""
    section("2. Reverse Ablation (Remove one external group)")

    log("Forward ablation (add one group) showed minimal effect.")
    log("Reverse ablation (remove one group from full model) checks")
    log("if any group's REMOVAL causes a significant drop.")
    log("")

    full_features = GROUP_A + GROUP_STORE_FULL + ALL_EXTERNAL

    r_full = train_xgb(train_df, test_df, full_features, "Model B (Full)")
    log(f"Model B (Full): AUROC={r_full['auroc']:.4f}")

    removals = [
        ("- Traffic", EXT_TRAFFIC),
        ("- Change Index", EXT_CHANGE),
        ("- Facilities", EXT_FACILITIES),
        ("- Macro (CSI)", EXT_MACRO),
        ("- Residents", EXT_RESIDENTS),
        ("- ALL External", ALL_EXTERNAL),
    ]

    log("")
    log("| Removed Group | AUROC | Delta vs Full |")
    log("|---|---|---|")

    for label, remove_group in removals:
        reduced = [f for f in full_features if f not in remove_group]
        r = train_xgb(train_df, test_df, reduced, f"B {label}")
        delta = r["auroc"] - r_full["auroc"]
        log(f"| {label} | {r['auroc']:.4f} | {'+' if delta >= 0 else ''}{delta:.4f} |")

    log("")
    log("If removing a group INCREASES AUROC, that group was adding noise.")
    log("If removing a group DECREASES AUROC, that group was contributing signal.")


# ================================================================
# Report Generator
# ================================================================
def generate_report():
    header = [
        "# Stage 4 Branch 2: Store Count Control + Reverse Ablation",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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
    print("Stage 4 Branch 2: Store Count Control + Reverse Ablation")
    print("=" * 60)

    train_df, test_df = load_and_split()

    store_control_experiment(train_df, test_df)
    reverse_ablation(train_df, test_df)

    generate_report()
    print("\nBranch 2 complete.")


if __name__ == "__main__":
    main()
