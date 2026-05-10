"""
Stage 4 Extended — Cold-Start, Segment Ablation, Rolling Validation, Threshold Sensitivity.

Core question: "External data shows no lift on average — but does it
matter in specific situations where card/sales data is limited?"

Experiments:
  1. Cold-start simulation: remove sales features, test external data value
  2. Segment ablation: low store_count, high-turnover industries, etc.
  3. Rolling time validation: confirm null result isn't period-specific
  4. Threshold sensitivity: confirm null result isn't threshold-specific

Usage: python run_extended_experiments.py
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
REPORT_PATH = "docs/extended_experiments.md"
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

LIMITED_SALES = ["total_sales", "total_txn_count", "sales_qoq_change"]

TARGET = "high_risk_flag"


# ================================================================
# Data Loading
# ================================================================
def load_data():
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    log(f"Loaded {len(df):,} rows")
    return df


def time_split(df, test_quarters=None):
    quarters = sorted(df["stdr_yyqu_cd"].unique())
    if test_quarters is None:
        test_quarters = quarters[-4:]
    train_quarters = [q for q in quarters if q not in test_quarters]
    train_df = df[df["stdr_yyqu_cd"].isin(train_quarters)].copy()
    test_df = df[df["stdr_yyqu_cd"].isin(test_quarters)].copy()
    return train_df, test_df


# ================================================================
# Model Training
# ================================================================
def train_xgb(train_df, test_df, feature_cols, target=TARGET):
    available = [c for c in feature_cols if c in train_df.columns]
    if len(available) == 0:
        return None

    X_train = train_df[available].copy()
    y_train = train_df[target].copy()
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "auroc": roc_auc_score(y_test, y_prob),
        "auprc": average_precision_score(y_test, y_prob),
        "n_features": len(available),
        "n_test": len(y_test),
        "pos_rate": y_test.mean(),
    }


# ================================================================
# 1. Cold-Start Simulation
# ================================================================
def cold_start_simulation(df):
    section("1. Cold-Start Simulation")

    log("When card/sales data is unavailable (new merchants, cash-heavy businesses),")
    log("does external data provide value as a substitute?")
    log("")

    train_df, test_df = time_split(df)

    scenarios = [
        ("Full (Sales+Store+External)", SALES_FEATURES + STORE_FEATURES + ALL_EXTERNAL),
        ("Sales+Store (no external)", SALES_FEATURES + STORE_FEATURES),
        ("Store+External (no sales)", STORE_FEATURES + ALL_EXTERNAL),
        ("Store only (no sales, no external)", STORE_FEATURES),
        ("Limited sales+Store+External", LIMITED_SALES + STORE_FEATURES + ALL_EXTERNAL),
        ("Limited sales+Store (no external)", LIMITED_SALES + STORE_FEATURES),
        ("External only (cold-start)", ALL_EXTERNAL),
    ]

    results = []
    for name, features in scenarios:
        r = train_xgb(train_df, test_df, features)
        if r:
            results.append({"Scenario": name, **r})
            log(f"{name}: AUROC={r['auroc']:.4f}, AUPRC={r['auprc']:.4f}")

    log("")
    log("### Cold-Start Results")
    log("")
    log("| Scenario | Features | AUROC | AUPRC |")
    log("|---|---|---|---|")
    for r in results:
        log(f"| {r['Scenario']} | {r['n_features']} | {r['auroc']:.4f} | {r['auprc']:.4f} |")

    # Key comparisons
    log("")
    log("### Key Comparisons")

    def get_auroc(name):
        r = next((x for x in results if x["Scenario"] == name), None)
        return r["auroc"] if r else None

    # No-sales: Store vs Store+External
    store_only = get_auroc("Store only (no sales, no external)")
    store_ext = get_auroc("Store+External (no sales)")
    if store_only and store_ext:
        delta = store_ext - store_only
        log(f"\n**No-sales scenario (cold-start proxy):**")
        log(f"  Store only: {store_only:.4f}")
        log(f"  Store + External: {store_ext:.4f}")
        log(f"  External lift: {'+' if delta >= 0 else ''}{delta:.4f}")
        if delta > 0.005:
            log(f"  -> External data DOES provide value when sales data is unavailable!")
        elif delta > 0.001:
            log(f"  -> Marginal value from external data in cold-start scenario.")
        else:
            log(f"  -> Even without sales data, external data adds minimal value.")

    # Limited sales: with vs without external
    ltd_only = get_auroc("Limited sales+Store (no external)")
    ltd_ext = get_auroc("Limited sales+Store+External")
    if ltd_only and ltd_ext:
        delta = ltd_ext - ltd_only
        log(f"\n**Limited-sales scenario:**")
        log(f"  Limited sales + Store: {ltd_only:.4f}")
        log(f"  Limited sales + Store + External: {ltd_ext:.4f}")
        log(f"  External lift: {'+' if delta >= 0 else ''}{delta:.4f}")

    # External only
    ext_only = get_auroc("External only (cold-start)")
    if ext_only:
        log(f"\n**Pure cold-start (external only):**")
        log(f"  AUROC: {ext_only:.4f}")
        log(f"  (Random baseline = 0.50)")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r["Scenario"] for r in results]
    aurocs = [r["auroc"] for r in results]

    colors = []
    for name in names:
        if "Full" in name:
            colors.append("#4682B4")
        elif "cold-start" in name.lower() or "External only" in name:
            colors.append("#E8744F")
        elif "no external" in name or "Store only" in name:
            colors.append("#CCCCCC")
        else:
            colors.append("#87CEEB")

    bars = ax.barh(range(len(results)), aurocs, color=colors, edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([n.replace("(", "\n(") for n in names], fontsize=8)
    ax.set_xlabel("AUROC")
    ax.set_title("Cold-Start Simulation: When Does External Data Matter?")
    ax.invert_yaxis()

    for i, v in enumerate(aurocs):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "model_14_cold_start.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"\n- Figure: {path}")

    return results


# ================================================================
# 2. Segment-Specific Ablation
# ================================================================
def segment_ablation(df):
    section("2. Segment-Specific Ablation")

    log("Does external data help more in specific segments?")
    log("")

    train_df, test_df = time_split(df)

    base_features = SALES_FEATURES + STORE_FEATURES
    full_features = base_features + ALL_EXTERNAL

    # Define segments
    segments = {
        "All (baseline)": (train_df, test_df),
    }

    # Low store_count (bottom 20%)
    threshold_store = df["store_count"].quantile(0.2)
    segments["store_count bottom 20%"] = (
        train_df[train_df["store_count"] <= threshold_store],
        test_df[test_df["store_count"] <= threshold_store],
    )

    # High store_count (top 20%)
    threshold_store_high = df["store_count"].quantile(0.8)
    segments["store_count top 20%"] = (
        train_df[train_df["store_count"] >= threshold_store_high],
        test_df[test_df["store_count"] >= threshold_store_high],
    )

    # High-turnover industries
    high_turnover = ["치킨전문점", "편의점", "패스트푸드점", "커피-음료", "분식전문점"]
    segments["High-turnover industries"] = (
        train_df[train_df["svc_induty_cd_nm"].isin(high_turnover)],
        test_df[test_df["svc_induty_cd_nm"].isin(high_turnover)],
    )

    # Stable industries
    stable = ["일반의원", "치과의원", "가전제품", "조명용품", "컴퓨터및주변장치판매"]
    segments["Stable industries"] = (
        train_df[train_df["svc_induty_cd_nm"].isin(stable)],
        test_df[test_df["svc_induty_cd_nm"].isin(stable)],
    )

    results = []
    for seg_name, (seg_train, seg_test) in segments.items():
        if len(seg_train) < 500 or len(seg_test) < 100:
            log(f"  {seg_name}: insufficient data ({len(seg_train)}/{len(seg_test)}), skipping")
            continue

        r_base = train_xgb(seg_train, seg_test, base_features)
        r_full = train_xgb(seg_train, seg_test, full_features)

        if r_base and r_full:
            delta_auroc = r_full["auroc"] - r_base["auroc"]
            delta_auprc = r_full["auprc"] - r_base["auprc"]
            results.append({
                "Segment": seg_name,
                "N_train": len(seg_train),
                "N_test": len(seg_test),
                "Pos_rate": seg_test[TARGET].mean(),
                "AUROC_base": r_base["auroc"],
                "AUROC_full": r_full["auroc"],
                "dAUROC": delta_auroc,
                "AUPRC_base": r_base["auprc"],
                "AUPRC_full": r_full["auprc"],
                "dAUPRC": delta_auprc,
            })

    log("### Segment Results")
    log("")
    log("| Segment | N_test | Pos% | AUROC(base) | AUROC(+ext) | dAUROC | dAUPRC |")
    log("|---|---|---|---|---|---|---|")
    for r in results:
        log(f"| {r['Segment']} | {r['N_test']:,} | {r['Pos_rate']:.1%} | "
            f"{r['AUROC_base']:.4f} | {r['AUROC_full']:.4f} | "
            f"{r['dAUROC']:+.4f} | {r['dAUPRC']:+.4f} |")

    # Find segments where external data helps most
    if results:
        best_seg = max(results, key=lambda x: x["dAUROC"])
        worst_seg = min(results, key=lambda x: x["dAUROC"])
        log(f"\nBest segment for external data: {best_seg['Segment']} (dAUROC={best_seg['dAUROC']:+.4f})")
        log(f"Worst segment: {worst_seg['Segment']} (dAUROC={worst_seg['dAUROC']:+.4f})")

    # Visualization
    if results:
        fig, ax = plt.subplots(figsize=(10, 5))
        seg_names = [r["Segment"] for r in results]
        deltas = [r["dAUROC"] for r in results]
        colors = ["#2E8B57" if d > 0 else "#E8744F" for d in deltas]

        ax.barh(range(len(results)), deltas, color=colors, edgecolor="black", alpha=0.8)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels(seg_names, fontsize=9)
        ax.set_xlabel("ΔAUROC (external data lift)")
        ax.set_title("External Data Lift by Segment")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        for i, v in enumerate(deltas):
            ax.text(v + 0.001 if v >= 0 else v - 0.001, i,
                    f"{v:+.4f}", va="center", fontsize=8,
                    ha="left" if v >= 0 else "right")

        plt.tight_layout()
        path = os.path.join(FIG_DIR, "model_15_segment_ablation.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        log(f"- Figure: {path}")

    return results


# ================================================================
# 3. Rolling Time Validation
# ================================================================
def rolling_validation(df):
    section("3. Rolling Time Validation")

    log("Confirm that the null result is not period-specific.")
    log("")

    quarters = sorted(df["stdr_yyqu_cd"].unique())
    base_features = SALES_FEATURES + STORE_FEATURES
    full_features = base_features + ALL_EXTERNAL

    # Rolling windows: each uses 4 test quarters
    windows = []
    for i in range(len(quarters) - 8, len(quarters) - 3):
        test_qs = quarters[i:i+4]
        if len(test_qs) == 4:
            windows.append(test_qs)

    results = []
    for test_qs in windows:
        train_df, test_df = time_split(df, test_quarters=test_qs)

        if len(train_df) < 1000 or test_df[TARGET].sum() < 50:
            continue

        r_base = train_xgb(train_df, test_df, base_features)
        r_full = train_xgb(train_df, test_df, full_features)

        if r_base and r_full:
            results.append({
                "Test Period": f"{test_qs[0]}~{test_qs[-1]}",
                "N_test": len(test_df),
                "AUROC_base": r_base["auroc"],
                "AUROC_full": r_full["auroc"],
                "dAUROC": r_full["auroc"] - r_base["auroc"],
            })

    log("| Test Period | N_test | AUROC(base) | AUROC(+ext) | dAUROC |")
    log("|---|---|---|---|---|")
    for r in results:
        log(f"| {r['Test Period']} | {r['N_test']:,} | "
            f"{r['AUROC_base']:.4f} | {r['AUROC_full']:.4f} | {r['dAUROC']:+.4f} |")

    if results:
        avg_delta = np.mean([r["dAUROC"] for r in results])
        all_negative = all(r["dAUROC"] <= 0.002 for r in results)
        log(f"\nAverage dAUROC across windows: {avg_delta:+.4f}")
        if all_negative:
            log("External data lift is consistently negligible across all time windows.")
        else:
            log("Some variation across windows detected.")

    return results


# ================================================================
# 4. Threshold Sensitivity
# ================================================================
def threshold_sensitivity(df):
    section("4. Threshold Sensitivity")

    log("Does the null result change with different high-risk thresholds?")
    log("")

    train_df_full, test_df_full = time_split(df)
    base_features = SALES_FEATURES + STORE_FEATURES
    full_features = base_features + ALL_EXTERNAL

    multipliers = [1.0, 1.25, 1.5, 2.0, 2.5]
    results = []

    for mult in multipliers:
        # Recompute target with different threshold
        industry_median = df.groupby("svc_induty_cd")["next_q_closure_rate"].transform("median")
        target_col = f"hr_{mult}"
        df[target_col] = (df["next_q_closure_rate"] > industry_median * mult).astype(int)

        train_df, test_df = time_split(df)
        pos_rate = test_df[target_col].mean()

        if pos_rate < 0.02 or pos_rate > 0.80:
            log(f"  Threshold {mult}x: pos_rate={pos_rate:.1%}, skipping (too extreme)")
            continue

        r_base = train_xgb(train_df, test_df, base_features, target=target_col)
        r_full = train_xgb(train_df, test_df, full_features, target=target_col)

        if r_base and r_full:
            results.append({
                "Threshold": f"{mult}x median",
                "Pos_rate": pos_rate,
                "AUROC_base": r_base["auroc"],
                "AUROC_full": r_full["auroc"],
                "dAUROC": r_full["auroc"] - r_base["auroc"],
            })

    log("| Threshold | Pos Rate | AUROC(base) | AUROC(+ext) | dAUROC |")
    log("|---|---|---|---|---|")
    for r in results:
        log(f"| {r['Threshold']} | {r['Pos_rate']:.1%} | "
            f"{r['AUROC_base']:.4f} | {r['AUROC_full']:.4f} | {r['dAUROC']:+.4f} |")

    if results:
        all_small = all(abs(r["dAUROC"]) < 0.005 for r in results)
        if all_small:
            log("\nNull result is robust across all threshold levels.")
        else:
            log("\nSome threshold sensitivity detected.")

    return results


# ================================================================
# Report
# ================================================================
def generate_report():
    header = [
        "# Stage 4 Extended: Cold-Start, Segments, Rolling, Threshold",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "When does external data matter? Does the null result hold universally?",
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
    print("Stage 4 Extended: Cold-Start + Segments + Rolling + Threshold")
    print("=" * 60)

    df = load_data()

    cold_start_simulation(df)

    segment_ablation(df)

    rolling_validation(df)

    threshold_sensitivity(df)

    generate_report()
    print("\nExtended experiments complete.")


if __name__ == "__main__":
    main()
