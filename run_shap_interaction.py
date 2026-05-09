"""
Stage 4 — Branch 1: SHAP Interaction + Divergence Features + Precision@K.

Completes plan sections 4.4 and 4.5:
  - SHAP Interaction: "매출 감소 × 유동인구 감소가 만나면 리스크가 얼마나 급등하는가"
  - Divergence features: signals that only emerge from crossing card + external data
  - Precision@K: "상위 K개 리스크 상권의 실제 폐업 비율"

Usage: python run_shap_interaction.py
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
REPORT_PATH = "docs/shap_interaction_results.md"
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
# Feature Groups (from run_model.py)
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

# Divergence features: signals from crossing card × external
DIVERGENCE_FEATURES = [
    "sales_traffic_divergence",
    "sales_up_traffic_down",
    "sales_down_traffic_up",
    "txn_vs_traffic_ratio",
    "sales_per_traffic",
]

TARGET = "high_risk_flag"


# ================================================================
# Data Loading + Divergence Feature Engineering
# ================================================================
def load_and_engineer():
    """Load data, create divergence features, split by time."""
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)

    log(f"Loaded {len(df):,} rows")

    # Divergence features: card × external crossing signals
    # 1. Sales QoQ - Traffic QoQ (directional divergence)
    df["sales_traffic_divergence"] = (
        df["sales_qoq_change"].fillna(0) - df["traffic_qoq_change"].fillna(0)
    )

    # 2. Binary: sales up but traffic down (people leaving but revenue up = unsustainable)
    df["sales_up_traffic_down"] = (
        (df["sales_qoq_change"] > 0) & (df["traffic_qoq_change"] < 0)
    ).astype(int)

    # 3. Binary: sales down but traffic up (people coming but not buying = trouble)
    df["sales_down_traffic_up"] = (
        (df["sales_qoq_change"] < 0) & (df["traffic_qoq_change"] > 0)
    ).astype(int)

    # 4. Transaction count / foot traffic ratio (conversion efficiency)
    df["txn_vs_traffic_ratio"] = np.where(
        df["total_foot_traffic"] > 0,
        df["total_txn_count"] / df["total_foot_traffic"],
        np.nan
    )

    # 5. Sales per foot traffic (revenue efficiency)
    df["sales_per_traffic"] = np.where(
        df["total_foot_traffic"] > 0,
        df["total_sales"] / df["total_foot_traffic"],
        np.nan
    )

    log(f"Divergence features created: {DIVERGENCE_FEATURES}")

    # Temporal split
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
    """Train XGBoost and return full result dict."""
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
        "y_prob": y_prob,
    }


# ================================================================
# 1. Precision@K
# ================================================================
def precision_at_k(y_true, y_prob, k_values=[50, 100, 200, 500]):
    """Calculate Precision@K: among top-K riskiest, how many actually closed?"""
    section("1. Precision@K Analysis")

    sorted_idx = np.argsort(-y_prob)
    results = {}

    log("| K | Precision@K | True Positives in Top-K | Base Rate |")
    log("|---|---|---|---|")

    base_rate = y_true.mean()
    for k in k_values:
        if k > len(y_true):
            continue
        top_k_labels = y_true.iloc[sorted_idx[:k]]
        prec_k = top_k_labels.mean()
        tp = int(top_k_labels.sum())
        results[k] = prec_k
        log(f"| {k} | {prec_k:.4f} | {tp}/{k} | {base_rate:.4f} |")

    log(f"\nInterpretation: Base rate = {base_rate:.4f}. "
        f"Precision@K > base rate means the model is concentrating risk correctly.")

    return results


# ================================================================
# 2. Divergence Feature Ablation
# ================================================================
def divergence_ablation(train_df, test_df):
    """Compare: base model vs base + divergence features."""
    section("2. Divergence Feature Ablation")

    base_features = GROUP_A + GROUP_STORE + ALL_EXTERNAL

    log("**Hypothesis:** Divergence features (card × external crossing signals) capture")
    log("information that neither card data nor external data provides independently.")
    log("")

    # Model B (no divergence)
    r_base = train_xgb(train_df, test_df, base_features, "Model B (No Divergence)")
    log(f"Model B (baseline): AUROC={r_base['auroc']:.4f}, F1={r_base['f1']:.4f}")

    # Model B + Divergence
    div_features = base_features + DIVERGENCE_FEATURES
    r_div = train_xgb(train_df, test_df, div_features, "Model B + Divergence")
    log(f"Model B + Divergence: AUROC={r_div['auroc']:.4f}, F1={r_div['f1']:.4f}")

    delta_auroc = r_div["auroc"] - r_base["auroc"]
    delta_f1 = r_div["f1"] - r_base["f1"]
    log(f"\nDelta AUROC: {'+' if delta_auroc >= 0 else ''}{delta_auroc:.4f}")
    log(f"Delta F1: {'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}")

    if delta_auroc > 0.001:
        log("\nConclusion: Divergence features provide incremental value.")
        log("The crossing of card and external signals contains information")
        log("that neither source provides alone.")
    else:
        log("\nConclusion: Divergence features do not meaningfully improve prediction.")
        log("Card data and external data operate on independent axes for this problem.")

    # Precision@K for both
    log("\n### Precision@K Comparison")
    log("")
    log("**Model B (No Divergence):**")
    pk_base = precision_at_k(r_base["y_test"], r_base["y_prob"])

    log("")
    log("**Model B + Divergence:**")
    pk_div = precision_at_k(r_div["y_test"], r_div["y_prob"])

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Model B\n(No Div)", "Model B\n(+ Divergence)"]
    aurocs = [r_base["auroc"], r_div["auroc"]]
    f1s = [r_base["f1"], r_div["f1"]]

    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, aurocs, w, label="AUROC", alpha=0.8, edgecolor="black")
    ax.bar(x + w/2, f1s, w, label="F1", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Divergence Feature Effect")
    ax.legend()

    for i, (a, f) in enumerate(zip(aurocs, f1s)):
        ax.text(i - w/2, a + 0.005, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + 0.005, f"{f:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_04_divergence.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"\n- Figure: {path}")

    return r_div


# ================================================================
# 3. SHAP Interaction Analysis
# ================================================================
def shap_interaction_analysis(result):
    """SHAP interaction values: which feature pairs amplify risk?"""
    section("3. SHAP Interaction Analysis")

    model = result["model"]
    X_test = result["X_test"]
    feature_names = result["feature_names"]

    # Sample for interaction (very slow on full dataset)
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)

    log(f"Computing SHAP interaction values on {sample_size} samples...")
    log("(This may take a few minutes)")

    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(X_sample)

    # Sum absolute interaction values across samples
    mean_interaction = np.abs(shap_interaction).mean(axis=0)
    np.fill_diagonal(mean_interaction, 0)  # Remove self-interaction

    # Top 10 interaction pairs
    n_features = len(feature_names)
    pairs = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            pairs.append((feature_names[i], feature_names[j], mean_interaction[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    log("\n### Top 15 Feature Interaction Pairs")
    log("")
    log("| Rank | Feature 1 | Feature 2 | Mean |Interaction| |")
    log("|---|---|---|---|")
    for rank, (f1, f2, val) in enumerate(pairs_sorted[:15], 1):
        # Mark card × external interactions
        f1_group = "Card" if f1 in GROUP_A else ("Store" if f1 in GROUP_STORE else "External")
        f2_group = "Card" if f2 in GROUP_A else ("Store" if f2 in GROUP_STORE else "External")
        cross = " *CROSS*" if f1_group != f2_group else ""
        log(f"| {rank} | {f1} ({f1_group}) | {f2} ({f2_group}) | {val:.4f} |{cross}")

    # Count cross-group interactions in top 15
    cross_count = sum(1 for f1, f2, _ in pairs_sorted[:15]
                      if (f1 in GROUP_A and f2 in ALL_EXTERNAL) or
                         (f2 in GROUP_A and f1 in ALL_EXTERNAL) or
                         (f1 in GROUP_STORE and f2 in ALL_EXTERNAL) or
                         (f2 in GROUP_STORE and f1 in ALL_EXTERNAL))

    log(f"\nCross-group interactions in top 15: {cross_count}/15")

    # Interaction heatmap (top 15 features)
    top_features_idx = []
    seen = set()
    for f1, f2, _ in pairs_sorted[:15]:
        if f1 not in seen:
            seen.add(f1)
            top_features_idx.append(feature_names.index(f1))
        if f2 not in seen:
            seen.add(f2)
            top_features_idx.append(feature_names.index(f2))
    top_features_idx = top_features_idx[:12]

    sub_matrix = mean_interaction[np.ix_(top_features_idx, top_features_idx)]
    sub_names = [feature_names[i] for i in top_features_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="YlOrRd")
    ax.set_xticks(range(len(sub_names)))
    ax.set_yticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=90, fontsize=8)
    ax.set_yticklabels(sub_names, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("SHAP Interaction Heatmap (Top Features)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_05_shap_interaction.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    # Key business question: sales_qoq × traffic_qoq interaction
    if "sales_qoq_change" in feature_names and "traffic_qoq_change" in feature_names:
        i_sales = feature_names.index("sales_qoq_change")
        i_traffic = feature_names.index("traffic_qoq_change")
        interaction_val = mean_interaction[i_sales, i_traffic]
        log(f"\n### Key Interaction: sales_qoq × traffic_qoq = {interaction_val:.4f}")
        log("This answers: 'When sales decline AND foot traffic decline coincide,")
        log("does the risk amplify beyond what each factor contributes alone?'")


# ================================================================
# Report Generator
# ================================================================
def generate_report():
    header = [
        "# Stage 4 Branch 1: SHAP Interaction + Divergence + Precision@K",
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
    print("Stage 4 Branch 1: SHAP Interaction + Divergence + Precision@K")
    print("=" * 60)

    train_df, test_df = load_and_engineer()

    result_div = divergence_ablation(train_df, test_df)

    shap_interaction_analysis(result_div)

    generate_report()
    print("\nBranch 1 complete.")


if __name__ == "__main__":
    main()
