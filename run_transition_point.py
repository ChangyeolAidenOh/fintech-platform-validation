"""
Stage 4 — Branch 3: Transition Point Prediction.

Instead of predicting "high closure rate", predict the TRANSITION:
"A district-industry that had ZERO closures now experiences its first closure."

This is a more actionable prediction — detecting the moment a healthy
trade area starts to deteriorate. External data may matter more here
because the transition is driven by environmental changes.

Usage: python run_transition_point.py
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
REPORT_PATH = "docs/transition_point_results.md"
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

GROUP_STORE = [
    "store_count", "franchise_ratio", "competition_density",
    "closure_rate", "store_count_qoq_change",
]

ALL_EXTERNAL = [
    "total_foot_traffic", "traffic_female_ratio", "traffic_young_adult_ratio",
    "traffic_qoq_change", "change_index_numeric", "operating_months_avg",
    "closed_months_avg", "subway_count", "bus_stop_count",
    "total_facility_count", "csi_avg", "total_residents",
    "total_households", "resident_young_adult_ratio", "resident_elderly_ratio",
]


# ================================================================
# Transition Point Target Construction
# ================================================================
def build_transition_dataset():
    """Build dataset where target = 'first closure event after zero-closure streak'."""
    section("1. Transition Point Dataset Construction")

    sql = """
    SELECT * FROM mart.risk_features
    ORDER BY trdar_cd, svc_induty_cd, stdr_yyqu_cd
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    log(f"Full dataset: {len(df):,} rows")

    # For each (district, industry), find rows where:
    # - current quarter closure_rate = 0 (no closures this quarter)
    # - next quarter has closures (next_q_closure_rate > 0)
    # This is the "transition point" — going from healthy to first closure

    # Also include rows where:
    # - current quarter closure_rate = 0
    # - next quarter closure_rate = 0 (stayed healthy — negative class)

    # Filter to rows where current closure_rate = 0
    zero_closure = df[df["closure_rate"] == 0].copy()
    log(f"Rows with zero current closure: {len(zero_closure):,} ({len(zero_closure)/len(df):.1%})")

    # Target: did closure happen next quarter?
    zero_closure["transition_flag"] = (zero_closure["next_q_closure_rate"] > 0).astype(int)

    transition_rate = zero_closure["transition_flag"].mean()
    log(f"Transition events: {zero_closure['transition_flag'].sum():,} ({transition_rate:.1%})")
    log(f"Non-transitions: {(zero_closure['transition_flag'] == 0).sum():,} ({1-transition_rate:.1%})")

    return zero_closure


def split_transition(df):
    """Temporal split for transition dataset."""
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    test_quarters = quarters[-4:]
    train_quarters = quarters[:-4]

    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()

    log(f"Train: {len(train_df):,} (transition rate: {train_df['transition_flag'].mean():.4f})")
    log(f"Test: {len(test_df):,} (transition rate: {test_df['transition_flag'].mean():.4f})")

    return train_df, test_df


# ================================================================
# Model Training
# ================================================================
def train_xgb(train_df, test_df, feature_cols, model_name, target="transition_flag"):
    available = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[available].copy()
    y_train = train_df[target].copy()
    X_test = test_df[available].copy()
    y_test = test_df[target].copy()

    if y_train.sum() == 0 or y_test.sum() == 0:
        log(f"  [WARN] {model_name}: insufficient positive samples")
        return None

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
# Transition Ablation
# ================================================================
def transition_ablation(train_df, test_df):
    """A vs B ablation on transition prediction."""
    section("2. Transition Point Ablation (A vs B)")

    log("Does external data matter MORE for transition prediction")
    log("than for general closure rate prediction?")
    log("")

    base_features = GROUP_A + GROUP_STORE

    experiments = [
        ("Model A (Card Only)", base_features),
        ("Model B (Card + All External)", base_features + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name)
        if r:
            results.append(r)
            log(f"{name}: AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}")

    if len(results) == 2:
        delta = results[1]["auroc"] - results[0]["auroc"]
        log(f"\nDelta AUROC (A→B): {'+' if delta >= 0 else ''}{delta:.4f}")

        if delta > 0.005:
            log("\n**Finding:** External data provides MORE value for transition prediction")
            log("than for general closure prediction. Environmental changes are early")
            log("signals of a trade area beginning to deteriorate.")
        else:
            log("\n**Finding:** External data does not provide more value for transition")
            log("prediction. The transition is driven by the same factors as general closure.")

    # Visualization
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = ["Model A\n(Card Only)", "Model B\n(Card + External)"]
        aurocs = [r["auroc"] for r in results]
        colors = ["steelblue", "forestgreen"]

        ax.bar(range(2), aurocs, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(range(2))
        ax.set_xticklabels(names)
        ax.set_ylabel("AUROC")
        ax.set_title("Transition Point Prediction: Card Only vs Card + External")

        for i, v in enumerate(aurocs):
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)

        plt.tight_layout()
        path = os.path.join(FIG_DIR, "model_08_transition_ablation.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"- Figure: {path}")

    return results


# ================================================================
# SHAP on Transition Model
# ================================================================
def transition_shap(result):
    """SHAP analysis on transition prediction model."""
    if result is None:
        return

    section("3. SHAP for Transition Prediction (Model B)")

    model = result["model"]
    X_test = result["X_test"]
    feature_names = result["feature_names"]

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
        elif feat in GROUP_STORE:
            group = "Store"
        else:
            group = "External"
        log(f"| {i+1} | {feat} | {row['mean_abs_shap']:.4f} | {group} |")

    # Count external in top 10
    top10_external = sum(1 for _, row in importance_df.head(10).iterrows()
                        if row["feature"] in ALL_EXTERNAL)
    log(f"\nExternal features in top 10: {top10_external}/10")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title("SHAP: Transition Point Prediction")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_09_transition_shap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# Report Generator
# ================================================================
def generate_report():
    header = [
        "# Stage 4 Branch 3: Transition Point Prediction",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Predicting the first closure event in previously healthy trade areas.",
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
    print("Stage 4 Branch 3: Transition Point Prediction")
    print("=" * 60)

    df = build_transition_dataset()
    train_df, test_df = split_transition(df)

    results = transition_ablation(train_df, test_df)

    # SHAP on Model B if available
    model_b = next((r for r in results if "Model B" in r["model_name"]), None) if results else None
    if model_b:
        transition_shap(model_b)

    generate_report()
    print("\nBranch 3 complete.")


if __name__ == "__main__":
    main()
