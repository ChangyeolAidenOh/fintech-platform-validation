"""
Stage 3 — Exploratory Data Analysis + Hypothesis Testing.
Analyzes mart.risk_features to validate assumptions before modeling.

Key questions:
  Q1: Which external features correlate with closure rate?
  Q2: Do different industries have different risk drivers?
  Q3: Preliminary signal strength of card-only vs external features

Outputs:
  - figures/eda_*.png (visualizations)
  - docs/exploratory_findings.md (report)

Usage: python run_eda.py
"""

# stdlib
import os
from datetime import datetime

# third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

# local
from database.connection import get_conn

load_dotenv()

FIG_DIR = "figures"
REPORT_PATH = "docs/exploratory_findings.md"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("docs", exist_ok=True)

plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
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
# Data Loading
# ================================================================
def load_risk_features():
    """Load mart.risk_features into DataFrame."""
    sql = "SELECT * FROM mart.risk_features"
    with get_conn() as conn:
        df = pd.read_sql(sql, conn)
    log(f"Loaded {len(df):,} rows from mart.risk_features")
    log(f"Columns: {len(df.columns)}")
    log(f"Quarters: {df['stdr_yyqu_cd'].nunique()} ({df['stdr_yyqu_cd'].min()} ~ {df['stdr_yyqu_cd'].max()})")
    log(f"Districts: {df['trdar_cd'].nunique()}")
    log(f"Industries: {df['svc_induty_cd'].nunique()}")
    return df


# ================================================================
# EDA 1: Target Variable Distribution
# ================================================================
def eda_target_distribution(df):
    """Analyze distribution of next_q_closure_rate."""
    section("1. Target Variable Distribution")

    target = df["next_q_closure_rate"].dropna()
    log(f"- Count: {len(target):,}")
    log(f"- Mean: {target.mean():.4f}")
    log(f"- Median: {target.median():.4f}")
    log(f"- Std: {target.std():.4f}")
    log(f"- Min: {target.min():.4f}, Max: {target.max():.4f}")
    log(f"- Zero closure rate: {(target == 0).sum():,} ({(target == 0).mean():.1%})")
    log(f"- High risk (flag=1): {df['high_risk_flag'].sum():,} ({df['high_risk_flag'].mean():.1%})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(target, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Next Quarter Closure Rate")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Next-Quarter Closure Rate")
    axes[0].axvline(target.median(), color="red", linestyle="--", label=f"Median={target.median():.4f}")
    axes[0].legend()

    # Capped at 0.3 for readability
    capped = target[target <= 0.3]
    axes[1].hist(capped, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Next Quarter Closure Rate (capped at 0.3)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution (Zoomed)")
    axes[1].axvline(target.median(), color="red", linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_01_target_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 2: Industry-Level Closure Rate Comparison
# ================================================================
def eda_industry_closure(df):
    """Compare closure rates across industries."""
    section("2. Industry-Level Closure Rate")

    industry_stats = (
        df.groupby("svc_induty_cd_nm")["next_q_closure_rate"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    industry_stats.columns = ["mean", "median", "std", "count"]

    log("Top 10 highest closure rate industries:")
    log("")
    log("| Industry | Mean | Median | Std | Count |")
    log("|---|---|---|---|---|")
    for idx, row in industry_stats.head(10).iterrows():
        log(f"| {idx} | {row['mean']:.4f} | {row['median']:.4f} | {row['std']:.4f} | {int(row['count']):,} |")

    log("")
    log("Bottom 5 lowest closure rate industries:")
    log("")
    log("| Industry | Mean | Median | Count |")
    log("|---|---|---|---|")
    for idx, row in industry_stats.tail(5).iterrows():
        log(f"| {idx} | {row['mean']:.4f} | {row['median']:.4f} | {int(row['count']):,} |")

    # Kruskal-Wallis test
    groups = [g["next_q_closure_rate"].dropna().values for _, g in df.groupby("svc_induty_cd")]
    groups = [g for g in groups if len(g) >= 10]
    stat_kw, p_kw = stats.kruskal(*groups)
    log(f"\nKruskal-Wallis test: H={stat_kw:.2f}, p={p_kw:.2e}")
    log(f"Conclusion: {'Industries have significantly different closure rates' if p_kw < 0.05 else 'No significant difference'}")

    # Plot top 15
    top15 = industry_stats.head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(top15)), top15["mean"], xerr=top15["std"], capsize=3, alpha=0.7)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=9)
    ax.set_xlabel("Mean Closure Rate")
    ax.set_title("Top 15 Industries by Mean Closure Rate")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_02_industry_closure.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 3: Sales QoQ vs Closure Rate
# ================================================================
def eda_sales_vs_closure(df):
    """Test if sales decline precedes closure."""
    section("3. Sales QoQ Change vs Closure Rate")

    valid = df[["sales_qoq_change", "next_q_closure_rate"]].dropna()
    # Clip extreme outliers for correlation
    valid = valid[
        (valid["sales_qoq_change"].between(-1, 5)) &
        (valid["next_q_closure_rate"] <= 0.5)
    ]

    corr, p_val = stats.spearmanr(valid["sales_qoq_change"], valid["next_q_closure_rate"])
    log(f"- Valid pairs: {len(valid):,}")
    log(f"- Spearman correlation: r={corr:.4f}, p={p_val:.2e}")
    log(f"- Interpretation: {'Negative correlation (sales decline -> higher closure)' if corr < 0 else 'No clear negative relationship'}")

    # Bin sales change into quartiles
    valid["sales_bin"] = pd.qcut(valid["sales_qoq_change"], q=5, labels=["Q1(lowest)", "Q2", "Q3", "Q4", "Q5(highest)"])
    bin_means = valid.groupby("sales_bin", observed=True)["next_q_closure_rate"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_means.plot(kind="bar", ax=ax, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Sales QoQ Change Quintile")
    ax.set_ylabel("Mean Next-Q Closure Rate")
    ax.set_title("Sales QoQ Change vs Next-Quarter Closure Rate")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_03_sales_vs_closure.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 4: Foot Traffic QoQ vs Closure Rate
# ================================================================
def eda_traffic_vs_closure(df):
    """Test if foot traffic decline correlates with closure."""
    section("4. Foot Traffic QoQ Change vs Closure Rate (External #1)")

    valid = df[["traffic_qoq_change", "next_q_closure_rate"]].dropna()
    valid = valid[
        (valid["traffic_qoq_change"].between(-1, 5)) &
        (valid["next_q_closure_rate"] <= 0.5)
    ]

    corr, p_val = stats.spearmanr(valid["traffic_qoq_change"], valid["next_q_closure_rate"])
    log(f"- Valid pairs: {len(valid):,}")
    log(f"- Spearman correlation: r={corr:.4f}, p={p_val:.2e}")

    valid["traffic_bin"] = pd.qcut(valid["traffic_qoq_change"], q=5, labels=["Q1(lowest)", "Q2", "Q3", "Q4", "Q5(highest)"],
                                   duplicates="drop")
    bin_means = valid.groupby("traffic_bin", observed=True)["next_q_closure_rate"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    bin_means.plot(kind="bar", ax=ax, alpha=0.7, edgecolor="black", color="teal")
    ax.set_xlabel("Foot Traffic QoQ Change Quintile")
    ax.set_ylabel("Mean Next-Q Closure Rate")
    ax.set_title("Foot Traffic QoQ Change vs Next-Quarter Closure Rate")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_04_traffic_vs_closure.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 5: District Change Index vs Closure Rate
# ================================================================
def eda_change_index_vs_closure(df):
    """Test if district change indicator relates to actual closure."""
    section("5. District Change Index vs Closure Rate (External #2)")

    valid = df[["change_index_name", "change_index_numeric", "next_q_closure_rate"]].dropna()

    group_stats = (
        valid.groupby("change_index_name")["next_q_closure_rate"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
    )

    log("| Change Index | Mean Closure | Median Closure | Count |")
    log("|---|---|---|---|")
    for idx, row in group_stats.iterrows():
        log(f"| {idx} | {row['mean']:.4f} | {row['median']:.4f} | {int(row['count']):,} |")

    # Chi-square: is change index associated with high_risk?
    valid_hr = df[["change_index_name", "high_risk_flag"]].dropna()
    if len(valid_hr) > 0:
        ct = pd.crosstab(valid_hr["change_index_name"], valid_hr["high_risk_flag"])
        chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        log(f"\nChi-square test (change_index vs high_risk): chi2={chi2:.2f}, p={p_chi:.2e}")
        log(f"Conclusion: {'Significant association' if p_chi < 0.05 else 'No significant association'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["다이나믹", "상권확장", "상권축소", "정체"]
    plot_data = group_stats.reindex([x for x in order if x in group_stats.index])
    plot_data["mean"].plot(kind="bar", ax=ax, alpha=0.7, edgecolor="black", color="coral")
    ax.set_xlabel("District Change Index")
    ax.set_ylabel("Mean Closure Rate")
    ax.set_title("District Change Index vs Closure Rate")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_05_change_index_vs_closure.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 6: Feature Correlation Heatmap
# ================================================================
def eda_correlation_heatmap(df):
    """Correlation of all numeric features with target."""
    section("6. Feature Correlation with Target")

    feature_cols = [
        # Group A: Card sales
        "total_sales", "total_txn_count", "avg_sales_per_txn",
        "weekend_sales_ratio", "female_sales_ratio", "young_adult_sales_ratio",
        "sales_qoq_change", "txn_count_qoq_change",
        # Group B: External
        "total_foot_traffic", "traffic_qoq_change",
        "change_index_numeric",
        "total_facility_count", "subway_count",
        "total_residents", "resident_young_adult_ratio", "resident_elderly_ratio",
        "csi_avg",
        # Store context
        "store_count", "franchise_ratio", "competition_density",
        "store_count_qoq_change",
        # Target
        "next_q_closure_rate",
    ]

    available = [c for c in feature_cols if c in df.columns]
    corr_df = df[available].corr(method="spearman")

    # Top correlations with target
    target_corr = corr_df["next_q_closure_rate"].drop("next_q_closure_rate").sort_values()
    log("Spearman correlations with next_q_closure_rate:")
    log("")
    log("| Feature | Correlation |")
    log("|---|---|")
    for feat, val in target_corr.items():
        marker = " ***" if abs(val) > 0.1 else ""
        log(f"| {feat} | {val:.4f}{marker} |")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_df.columns, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap (Spearman)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_06_correlation_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# EDA 7: Missing Data Summary
# ================================================================
def eda_missing_data(df):
    """Check missing data patterns."""
    section("7. Missing Data Summary")

    missing = df.isnull().sum()
    missing_pct = df.isnull().mean()
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("pct", ascending=False)

    if len(missing_df) == 0:
        log("No missing values found.")
    else:
        log("| Column | Missing | Pct |")
        log("|---|---|---|")
        for idx, row in missing_df.iterrows():
            log(f"| {idx} | {int(row['missing']):,} | {row['pct']:.1%} |")


# ================================================================
# EDA 8: Temporal Trend
# ================================================================
def eda_temporal_trend(df):
    """Closure rate trend over quarters."""
    section("8. Temporal Trend")

    quarterly = (
        df.groupby("stdr_yyqu_cd")
        .agg(
            avg_closure=("next_q_closure_rate", "mean"),
            high_risk_pct=("high_risk_flag", "mean"),
            count=("next_q_closure_rate", "count"),
        )
        .sort_index()
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(quarterly.index, quarterly["avg_closure"], marker="o", color="tab:red", label="Avg Closure Rate")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Avg Closure Rate", color="tab:red")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(quarterly.index, quarterly["count"], alpha=0.2, color="tab:blue", label="Row Count")
    ax2.set_ylabel("Row Count", color="tab:blue")

    ax1.set_title("Closure Rate Trend Over Time")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "eda_08_temporal_trend.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log(f"- Figure: {path}")


# ================================================================
# Report Generator
# ================================================================
def generate_report():
    """Write markdown report."""
    header = [
        "# Stage 3: Exploratory Findings",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Source:** mart.risk_features",
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
    print("Stage 3: Exploratory Data Analysis")
    print("=" * 60)

    df = load_risk_features()

    eda_target_distribution(df)
    eda_industry_closure(df)
    eda_sales_vs_closure(df)
    eda_traffic_vs_closure(df)
    eda_change_index_vs_closure(df)
    eda_correlation_heatmap(df)
    eda_missing_data(df)
    eda_temporal_trend(df)

    generate_report()
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
