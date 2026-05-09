"""
Stage 4 — Leakage Verification + Naive Baseline.

Critical checks:
  1. Naive Baseline: "previous quarter closure_rate predicts next quarter"
     If this matches XGBoost, the model learned nothing beyond inertia.
  2. closure_rate removal: does performance collapse without it?
  3. Strict lag-only features: remove ALL same-quarter store outcome variables
  4. Industry-naive baseline: just predict by industry average

Usage: python run_leakage_check.py
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
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()
warnings.filterwarnings("ignore")

FIG_DIR = "figures"
REPORT_PATH = "docs/leakage_check_results.md"
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


TARGET = "high_risk_flag"


# ================================================================
# Feature Group Definitions
# ================================================================

# Potentially leaky: same-quarter store outcome variables
LEAKY_SUSPECTS = [
    "closure_rate",       # current quarter closure rate
    "close_count",        # current quarter close count
    "open_count",         # current quarter open count
    "net_store_change",   # current quarter net change
]

# Card sales features (no leakage concern)
GROUP_A_CLEAN = [
    "total_sales", "total_txn_count", "avg_sales_per_txn",
    "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
    "sales_qoq_change", "txn_count_qoq_change", "sales_vs_2q_ma",
]

# Store context - STRICT (only structural, not outcome)
GROUP_STORE_STRICT = [
    "store_count",
    "franchise_ratio",
    "competition_density",
    "store_count_qoq_change",
]

# Store context - with leaky vars
GROUP_STORE_WITH_LEAKY = GROUP_STORE_STRICT + ["closure_rate"]

# External features
ALL_EXTERNAL = [
    "total_foot_traffic", "traffic_female_ratio", "traffic_young_adult_ratio",
    "traffic_qoq_change", "change_index_numeric", "operating_months_avg",
    "closed_months_avg", "subway_count", "bus_stop_count",
    "total_facility_count", "csi_avg", "total_residents",
    "total_households", "resident_young_adult_ratio", "resident_elderly_ratio",
]


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

    log(f"Train: {len(train_df):,} ({train_quarters[0]}~{train_quarters[-1]})")
    log(f"Test: {len(test_df):,} ({test_quarters[0]}~{test_quarters[-1]})")
    log(f"Train target rate: {train_df[TARGET].mean():.4f}")
    log(f"Test target rate: {test_df[TARGET].mean():.4f}")
    log(f"Time-based split: train ends at {train_quarters[-1]}, test starts at {test_quarters[0]}")
    log(f"Same district-industry combos in both: {len(set(zip(train_df['trdar_cd'], train_df['svc_induty_cd'])) & set(zip(test_df['trdar_cd'], test_df['svc_induty_cd'])))}")

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
        "feature_list": available,
        "auroc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "y_test": y_test,
        "y_prob": y_prob,
    }


def precision_at_k(y_true, y_prob, k_values=[50, 100, 200, 500]):
    sorted_idx = np.argsort(-y_prob)
    results = {}
    for k in k_values:
        if k > len(y_true):
            continue
        top_k_labels = y_true.iloc[sorted_idx[:k]]
        results[k] = top_k_labels.mean()
    return results


# ================================================================
# 1. Naive Baselines
# ================================================================
def naive_baselines(train_df, test_df):
    section("1. Naive Baselines (How hard is this problem really?)")

    log("If naive baselines match XGBoost, the model learned nothing meaningful.")
    log("")

    y_test = test_df[TARGET].copy()
    results = []

    # Baseline 1: Previous quarter closure_rate as score
    log("### Baseline 1: Previous Quarter Closure Rate")
    score = test_df["closure_rate"].fillna(0).copy()
    try:
        auroc = roc_auc_score(y_test, score)
        pk = precision_at_k(y_test, score.values)
        log(f"  AUROC: {auroc:.4f}")
        log(f"  Precision@100: {pk.get(100, 'N/A'):.4f}")
        log(f"  Precision@500: {pk.get(500, 'N/A'):.4f}")
        results.append(("Naive: prev closure_rate", auroc, pk.get(100, 0)))
    except Exception as e:
        log(f"  Error: {e}")

    # Baseline 2: Industry average closure rate
    log("")
    log("### Baseline 2: Industry Average Closure Rate")
    industry_avg = train_df.groupby("svc_induty_cd")["next_q_closure_rate"].mean()
    score2 = test_df["svc_induty_cd"].map(industry_avg).fillna(0)
    try:
        auroc2 = roc_auc_score(y_test, score2)
        pk2 = precision_at_k(y_test, score2.values)
        log(f"  AUROC: {auroc2:.4f}")
        log(f"  Precision@100: {pk2.get(100, 'N/A'):.4f}")
        log(f"  Precision@500: {pk2.get(500, 'N/A'):.4f}")
        results.append(("Naive: industry avg", auroc2, pk2.get(100, 0)))
    except Exception as e:
        log(f"  Error: {e}")

    # Baseline 3: store_count only
    log("")
    log("### Baseline 3: store_count Only")
    score3 = test_df["store_count"].fillna(0).copy()
    try:
        auroc3 = roc_auc_score(y_test, score3)
        pk3 = precision_at_k(y_test, score3.values)
        log(f"  AUROC: {auroc3:.4f}")
        log(f"  Precision@100: {pk3.get(100, 'N/A'):.4f}")
        log(f"  Precision@500: {pk3.get(500, 'N/A'):.4f}")
        results.append(("Naive: store_count only", auroc3, pk3.get(100, 0)))
    except Exception as e:
        log(f"  Error: {e}")

    # Baseline 4: closure_rate only (single feature XGBoost)
    log("")
    log("### Baseline 4: closure_rate Only (XGBoost, 1 feature)")
    r4 = train_xgb(train_df, test_df, ["closure_rate"], "XGB: closure_rate only")
    pk4 = precision_at_k(r4["y_test"], r4["y_prob"])
    log(f"  AUROC: {r4['auroc']:.4f}")
    log(f"  Precision@100: {pk4.get(100, 'N/A'):.4f}")
    log(f"  Precision@500: {pk4.get(500, 'N/A'):.4f}")
    results.append(("XGB: closure_rate only", r4["auroc"], pk4.get(100, 0)))

    return results


# ================================================================
# 2. Leakage Test: With vs Without closure_rate
# ================================================================
def leakage_test(train_df, test_df):
    section("2. Leakage Test: closure_rate Impact")

    log("closure_rate = current quarter's closure count / store count.")
    log("Target = NEXT quarter's closure rate (LEAD window).")
    log("Time separation exists, but autocorrelation may inflate performance.")
    log("")

    experiments = [
        ("Full (with closure_rate)",
         GROUP_A_CLEAN + GROUP_STORE_WITH_LEAKY),
        ("Strict (without closure_rate)",
         GROUP_A_CLEAN + GROUP_STORE_STRICT),
        ("Full + External (with closure_rate)",
         GROUP_A_CLEAN + GROUP_STORE_WITH_LEAKY + ALL_EXTERNAL),
        ("Strict + External (without closure_rate)",
         GROUP_A_CLEAN + GROUP_STORE_STRICT + ALL_EXTERNAL),
    ]

    results = []
    for name, features in experiments:
        r = train_xgb(train_df, test_df, features, name)
        pk = precision_at_k(r["y_test"], r["y_prob"])
        results.append({**r, "pk100": pk.get(100, 0), "pk500": pk.get(500, 0)})
        log(f"{name}:")
        log(f"  AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}, P@100={pk.get(100,0):.4f}, P@500={pk.get(500,0):.4f}")

    log("")
    log("### Leakage Impact Summary")
    log("")
    log("| Model | Features | AUROC | F1 | P@100 | P@500 |")
    log("|---|---|---|---|---|---|")
    for r in results:
        log(f"| {r['model_name']} | {r['features']} | {r['auroc']:.4f} | {r['f1']:.4f} | {r['pk100']:.4f} | {r['pk500']:.4f} |")

    # Key comparison
    with_leak = next(r for r in results if r["model_name"] == "Full (with closure_rate)")
    without_leak = next(r for r in results if r["model_name"] == "Strict (without closure_rate)")
    delta = with_leak["auroc"] - without_leak["auroc"]

    log(f"\n### closure_rate contribution: AUROC delta = {delta:.4f}")

    if delta > 0.05:
        log("**WARNING: closure_rate contributes significantly.**")
        log("This suggests autocorrelation-driven inflation.")
        log("The 'strict' model without closure_rate is the honest benchmark.")
    elif delta > 0.01:
        log("**CAUTION: closure_rate has moderate contribution.**")
        log("Some performance inflation from autocorrelation likely.")
    else:
        log("closure_rate contribution is minimal. No significant leakage concern.")

    return results


# ================================================================
# 3. Strict Model: Card-Only vs Card+External (no leaky vars)
# ================================================================
def strict_ablation(train_df, test_df):
    section("3. Strict Ablation (All Leaky Variables Removed)")

    log("This is the HONEST comparison: no closure_rate, no close_count,")
    log("no open_count, no net_store_change in features.")
    log("Only structural + behavioral features that are truly available BEFORE the prediction quarter.")
    log("")

    strict_card = GROUP_A_CLEAN + GROUP_STORE_STRICT
    strict_full = GROUP_A_CLEAN + GROUP_STORE_STRICT + ALL_EXTERNAL

    r_card = train_xgb(train_df, test_df, strict_card, "Strict A (Card Only)")
    r_full = train_xgb(train_df, test_df, strict_full, "Strict B (Card + External)")

    pk_card = precision_at_k(r_card["y_test"], r_card["y_prob"])
    pk_full = precision_at_k(r_full["y_test"], r_full["y_prob"])

    delta_auroc = r_full["auroc"] - r_card["auroc"]
    delta_f1 = r_full["f1"] - r_card["f1"]

    log(f"Strict A (Card Only):     AUROC={r_card['auroc']:.4f}, F1={r_card['f1']:.4f}, P@100={pk_card.get(100,0):.4f}")
    log(f"Strict B (Card+External): AUROC={r_full['auroc']:.4f}, F1={r_full['f1']:.4f}, P@100={pk_full.get(100,0):.4f}")
    log(f"Delta AUROC: {'+' if delta_auroc >= 0 else ''}{delta_auroc:.4f}")
    log(f"Delta F1: {'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}")

    log("")
    if delta_auroc > 0.005:
        log("**NEW FINDING: With leaky variables removed, external data DOES provide value.**")
        log("Previous null result was masked by closure_rate's dominance.")
    elif delta_auroc > 0.001:
        log("Marginal improvement from external data after leakage removal.")
    else:
        log("Null result holds even after removing leaky variables.")
        log("External public data genuinely adds no incremental value at this resolution.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Naive vs XGBoost comparison
    models = ["Naive:\nprev closure_rate", "Naive:\nindustry avg", "Naive:\nstore_count",
              "XGB:\nclosure_rate only",
              "XGB Strict:\nCard Only", "XGB Strict:\nCard+External"]
    # We'll fill these in main()

    # Right: With vs Without closure_rate
    names = ["With\nclosure_rate", "Without\nclosure_rate"]
    aurocs = [
        next(r for r in leakage_results if "Full (with" in r["model_name"])["auroc"],
        next(r for r in leakage_results if "Strict (without" in r["model_name"])["auroc"],
    ] if 'leakage_results' in dir() else [0, 0]

    return r_card, r_full


# ================================================================
# 4. Comprehensive Comparison Chart
# ================================================================
def comprehensive_chart(naive_results, leakage_results, strict_card, strict_full):
    section("4. Comprehensive Model Comparison")

    all_results = []

    # Naive baselines
    for name, auroc, pk100 in naive_results:
        all_results.append({"name": name, "auroc": auroc, "pk100": pk100, "category": "Naive"})

    # Leakage test results
    for r in leakage_results:
        pk = precision_at_k(r["y_test"], r["y_prob"])
        all_results.append({
            "name": f"XGB: {r['model_name']}",
            "auroc": r["auroc"],
            "pk100": pk.get(100, 0),
            "category": "XGB (with leaky)"
        })

    # Strict results
    pk_card = precision_at_k(strict_card["y_test"], strict_card["y_prob"])
    pk_full = precision_at_k(strict_full["y_test"], strict_full["y_prob"])
    all_results.append({
        "name": "XGB Strict: Card Only",
        "auroc": strict_card["auroc"],
        "pk100": pk_card.get(100, 0),
        "category": "XGB (strict)"
    })
    all_results.append({
        "name": "XGB Strict: Card+External",
        "auroc": strict_full["auroc"],
        "pk100": pk_full.get(100, 0),
        "category": "XGB (strict)"
    })

    # Table
    log("")
    log("| Category | Model | AUROC | P@100 |")
    log("|---|---|---|---|")
    for r in all_results:
        log(f"| {r['category']} | {r['name']} | {r['auroc']:.4f} | {r['pk100']:.4f} |")

    # Chart
    fig, ax = plt.subplots(figsize=(14, 7))

    names = [r["name"] for r in all_results]
    aurocs = [r["auroc"] for r in all_results]
    categories = [r["category"] for r in all_results]

    color_map = {
        "Naive": "lightgray",
        "XGB (with leaky)": "salmon",
        "XGB (strict)": "steelblue",
    }
    colors = [color_map.get(c, "gray") for c in categories]

    bars = ax.barh(range(len(all_results)), aurocs, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(all_results)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUROC")
    ax.set_title("Comprehensive Model Comparison: Naive vs XGB (Leaky vs Strict)")
    ax.invert_yaxis()

    for i, v in enumerate(aurocs):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgray", edgecolor="black", label="Naive Baseline"),
        Patch(facecolor="salmon", edgecolor="black", label="XGB (with closure_rate)"),
        Patch(facecolor="steelblue", edgecolor="black", label="XGB (strict, no leakage)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_10_comprehensive.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")

    # Key takeaway
    naive_best = max(naive_results, key=lambda x: x[1])
    strict_best = max([strict_card["auroc"], strict_full["auroc"]])

    log(f"\n### Key Takeaway")
    log(f"- Best naive baseline: {naive_best[0]} (AUROC={naive_best[1]:.4f})")
    log(f"- Best strict XGBoost: AUROC={strict_best:.4f}")
    log(f"- XGBoost improvement over best naive: {strict_best - naive_best[1]:.4f}")

    if strict_best - naive_best[1] < 0.02:
        log("\n**WARNING: XGBoost barely beats naive baselines.**")
        log("Most of the 'prediction' is autocorrelation / industry pattern.")
    elif strict_best - naive_best[1] < 0.05:
        log("\nXGBoost provides moderate improvement over naive baselines.")
    else:
        log("\nXGBoost provides substantial improvement over naive baselines.")


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Stage 4: Leakage Verification + Naive Baseline Check",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Critical validation: Is the model actually learning, or just exploiting autocorrelation?",
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
    print("Stage 4: Leakage Verification + Naive Baseline")
    print("=" * 60)

    train_df, test_df = load_and_split()

    naive_results = naive_baselines(train_df, test_df)

    leakage_results = leakage_test(train_df, test_df)

    strict_card, strict_full = strict_ablation(train_df, test_df)

    comprehensive_chart(naive_results, leakage_results, strict_card, strict_full)

    generate_report()
    print("\nLeakage check complete.")


if __name__ == "__main__":
    main()
