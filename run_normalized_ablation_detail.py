"""
Stage 4 — Normalized Target: Individual External Group Ablation.

SHAP says external data = 30.5% contribution on normalized target.
But AUROC improvement = +0.0012. Why the gap?

Hypothesis A: CSI alone drives the 30.5%, other externals are noise.
Hypothesis B: External features are redundant with card features (same signal, different angle).

Test: Add each external group individually to card-only model on normalized target.
If CSI alone lifts AUROC/AUPRC, Hypothesis A is confirmed.
If nothing lifts, Hypothesis B is confirmed (SHAP contribution ≠ predictive improvement).

Usage: python run_normalized_ablation_detail.py
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
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/normalized_ablation_detail.md"
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
GROUP_A_CLEAN = [
    "total_sales", "total_txn_count", "avg_sales_per_txn",
    "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
    "sales_qoq_change", "txn_count_qoq_change", "sales_vs_2q_ma",
]

GROUP_STORE_STRICT = [
    "store_count", "franchise_ratio", "competition_density",
    "store_count_qoq_change",
]

GROUP_STORE_NO_SIZE = [
    "franchise_ratio", "competition_density",
    "store_count_qoq_change",
]

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

TARGET = "excess_high_risk"


# ================================================================
# Data Loading
# ================================================================
def load_and_normalize():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    industry_q_avg = (
        df.groupby(["stdr_yyqu_cd", "svc_induty_cd"])["next_q_closure_rate"]
        .transform("mean")
    )
    df["excess_closure_rate"] = df["next_q_closure_rate"] - industry_q_avg
    df["excess_high_risk"] = (df["excess_closure_rate"] > 0).astype(int)

    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]

    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()

    log(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    log(f"Target rate: {test_df[TARGET].mean():.4f}")

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

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    return {
        "model_name": model_name,
        "features": len(available),
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
    }


# ================================================================
# 1. Individual External Group Ablation (With store_count)
# ================================================================
def individual_ablation_with_size(train_df, test_df):
    section("1. Individual External Group Ablation (Normalized Target, With store_count)")

    base = GROUP_A_CLEAN + GROUP_STORE_STRICT

    experiments = [
        ("A: Card + Store (baseline)", base),
        ("+ Traffic", base + EXT_TRAFFIC),
        ("+ Change Index", base + EXT_CHANGE),
        ("+ Facilities", base + EXT_FACILITIES),
        ("+ CSI (Macro)", base + EXT_MACRO),
        ("+ Residents", base + EXT_RESIDENTS),
        ("+ ALL External", base + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name)
        results.append(r)

    baseline = results[0]

    log("| Model | AUROC | AUPRC | dAUROC | dAUPRC |")
    log("|---|---|---|---|---|")
    for r in results:
        da = r["auroc"] - baseline["auroc"]
        dp = r["auprc"] - baseline["auprc"]
        da_str = f"{'+' if da >= 0 else ''}{da:.4f}" if r != baseline else "—"
        dp_str = f"{'+' if dp >= 0 else ''}{dp:.4f}" if r != baseline else "—"
        log(f"| {r['model_name']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {da_str} | {dp_str} |")

    return results


# ================================================================
# 2. Individual External Group Ablation (Without store_count)
# ================================================================
def individual_ablation_no_size(train_df, test_df):
    section("2. Individual External Group Ablation (Normalized Target, No store_count)")

    log("Full size-effect removal. This is where external data should shine if it has value.")
    log("")

    base = GROUP_A_CLEAN + GROUP_STORE_NO_SIZE

    experiments = [
        ("A: Card + Store/noSize (baseline)", base),
        ("+ Traffic", base + EXT_TRAFFIC),
        ("+ Change Index", base + EXT_CHANGE),
        ("+ Facilities", base + EXT_FACILITIES),
        ("+ CSI (Macro)", base + EXT_MACRO),
        ("+ Residents", base + EXT_RESIDENTS),
        ("+ ALL External", base + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name)
        results.append(r)

    baseline = results[0]

    log("| Model | AUROC | AUPRC | dAUROC | dAUPRC |")
    log("|---|---|---|---|---|")
    for r in results:
        da = r["auroc"] - baseline["auroc"]
        dp = r["auprc"] - baseline["auprc"]
        da_str = f"{'+' if da >= 0 else ''}{da:.4f}" if r != baseline else "—"
        dp_str = f"{'+' if dp >= 0 else ''}{dp:.4f}" if r != baseline else "—"
        log(f"| {r['model_name']} | {r['auroc']:.4f} | {r['auprc']:.4f} | {da_str} | {dp_str} |")

    # Find best individual
    individuals = results[1:-1]  # exclude baseline and ALL
    best = max(individuals, key=lambda x: x["auroc"])
    best_da = best["auroc"] - baseline["auroc"]
    best_dp = best["auprc"] - baseline["auprc"]

    log(f"\n### Best individual external group: {best['model_name']}")
    log(f"  AUROC lift: {'+' if best_da >= 0 else ''}{best_da:.4f}")
    log(f"  AUPRC lift: {'+' if best_dp >= 0 else ''}{best_dp:.4f}")

    return results, baseline


# ================================================================
# 3. Interpretation
# ================================================================
def interpret(results_size, results_nosize, baseline_nosize):
    section("3. Interpretation: SHAP 30.5% vs AUROC +0.001")

    # Find CSI results in no-size experiments
    csi_nosize = next((r for r in results_nosize if "CSI" in r["model_name"]), None)

    if csi_nosize:
        da = csi_nosize["auroc"] - baseline_nosize["auroc"]
        dp = csi_nosize["auprc"] - baseline_nosize["auprc"]

        log(f"CSI alone (no store_count, normalized target):")
        log(f"  AUROC lift: {'+' if da >= 0 else ''}{da:.4f}")
        log(f"  AUPRC lift: {'+' if dp >= 0 else ''}{dp:.4f}")
        log("")

        if da > 0.005 or dp > 0.005:
            log("**Hypothesis A confirmed: CSI is the primary driver of external value.**")
            log("Consumer sentiment is a genuine leading indicator of excess closure.")
            log("Other external data (traffic, facilities, residents) adds noise.")
        elif da > 0.001 or dp > 0.001:
            log("**Partial support for Hypothesis A.**")
            log("CSI provides weak but measurable signal for excess closure prediction.")
        else:
            log("**Hypothesis B confirmed: SHAP contribution ≠ predictive improvement.**")
            log("External features redistribute SHAP credit but don't add new information.")
            log("They capture the same underlying signal as card/store features from a different angle.")

    log("")
    log("### Why SHAP 30.5% but AUROC +0.001?")
    log("")
    log("SHAP measures how much the model USES a feature for predictions.")
    log("AUROC measures whether ADDING that feature improves DISCRIMINATIVE ability.")
    log("These are different questions:")
    log("- If external features are correlated with card features,")
    log("  the model will USE them (SHAP > 0) but they won't ADD new info (AUROC ~ 0).")
    log("- This is feature redundancy, not uselessness.")
    log("- In practice, this means: if card data is unavailable, external data")
    log("  could serve as a partial substitute — but it doesn't enhance card data.")

    # Visualization
    all_results = results_nosize
    baseline = all_results[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: AUROC deltas
    names = [r["model_name"].replace("A: Card + Store/noSize (baseline)", "Baseline") for r in all_results]
    auroc_deltas = [r["auroc"] - baseline["auroc"] for r in all_results]
    auprc_deltas = [r["auprc"] - baseline["auprc"] for r in all_results]

    x = np.arange(len(all_results))
    w = 0.35
    axes[0].bar(x - w/2, auroc_deltas, w, label="dAUROC", alpha=0.8, edgecolor="black", color="steelblue")
    axes[0].bar(x + w/2, auprc_deltas, w, label="dAUPRC", alpha=0.8, edgecolor="black", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=7, rotation=30, ha="right")
    axes[0].set_ylabel("Delta vs Baseline")
    axes[0].set_title("Normalized Target: Individual External Group Lift (No store_count)")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend()

    # Right: SHAP vs AUROC comparison
    shap_pcts = [36.6, 32.9, 30.5]  # from previous experiment
    auroc_pcts_data = []

    # Compute relative AUROC contribution
    a_only = next(r for r in results_nosize if "baseline" in r["model_name"])
    b_all = next(r for r in results_nosize if "ALL" in r["model_name"])

    axes[1].bar([0, 1, 2], shap_pcts, alpha=0.8, edgecolor="black",
               color=["lightsteelblue", "steelblue", "coral"],
               label="SHAP Contribution %")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Store", "Card", "External"])
    axes[1].set_ylabel("SHAP Contribution %")
    axes[1].set_title("SHAP Says 30.5% External\nBut AUROC Lift ≈ 0")
    for i, v in enumerate(shap_pcts):
        axes[1].text(i, v + 0.5, f"{v}%", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_13_normalized_individual.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Stage 4: Normalized Target — Individual External Group Ablation",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Resolving the SHAP 30.5% vs AUROC +0.001 paradox.",
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
    print("Stage 4: Normalized Target — Individual External Group Detail")
    print("=" * 60)

    train_df, test_df = load_and_normalize()

    results_size = individual_ablation_with_size(train_df, test_df)
    results_nosize, baseline_nosize = individual_ablation_no_size(train_df, test_df)

    interpret(results_size, results_nosize, baseline_nosize)

    generate_report()
    print("\nDone.")


if __name__ == "__main__":
    main()
