"""
Exploratory Data Analysis -- Phase 2

Generates statistical summaries and visualizations from the cleaned dataset,
saving all plots to reports/figures/.

Usage:
    python -m src.eda
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PATHS, PROJECT_ROOT
from src.logger import get_logger
from src.preprocessing import run_preprocessing
from src.utils import load_parquet

logger = get_logger(__name__)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _save_fig(fig, name: str) -> None:
    """Save a figure and close it."""
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved -> %s", path.name)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_sales_distribution(df: pd.DataFrame) -> None:
    """Histogram of Weekly_Sales."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Weekly_Sales"], bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.set_title("Distribution of Weekly Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weekly Sales ($)")
    ax.set_ylabel("Frequency")
    ax.axvline(df["Weekly_Sales"].mean(), color="red", linestyle="--", label=f'Mean: ${df["Weekly_Sales"].mean():,.0f}')
    ax.axvline(df["Weekly_Sales"].median(), color="orange", linestyle="--", label=f'Median: ${df["Weekly_Sales"].median():,.0f}')
    ax.legend()
    _save_fig(fig, "01_sales_distribution.png")


def plot_sales_over_time(df: pd.DataFrame) -> None:
    """Average weekly sales over time (all stores)."""
    weekly_avg = df.groupby("Date")["Weekly_Sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weekly_avg["Date"], weekly_avg["Weekly_Sales"], color="#4CAF50", linewidth=1.5)
    ax.set_title("Average Weekly Sales Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "02_sales_over_time.png")


def plot_sales_by_store(df: pd.DataFrame) -> None:
    """Bar chart of average sales per store."""
    store_avg = df.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#FF5722" if v == store_avg.max() else "#90CAF9" for v in store_avg.values]
    ax.bar(store_avg.index.astype(str), store_avg.values, color=colors, edgecolor="white")
    ax.set_title("Average Weekly Sales by Store", fontsize=14, fontweight="bold")
    ax.set_xlabel("Store")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, "03_sales_by_store.png")


def plot_holiday_impact(df: pd.DataFrame) -> None:
    """Box plot comparing holiday vs non-holiday sales."""
    fig, ax = plt.subplots(figsize=(8, 5))
    holiday_data = [
        df[df["Holiday_Flag"] == 0]["Weekly_Sales"],
        df[df["Holiday_Flag"] == 1]["Weekly_Sales"],
    ]
    bp = ax.boxplot(holiday_data, labels=["Non-Holiday", "Holiday"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#90CAF9")
    bp["boxes"][1].set_facecolor("#FF8A65")
    ax.set_title("Weekly Sales: Holiday vs Non-Holiday", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weekly Sales ($)")
    _save_fig(fig, "04_holiday_impact.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation matrix of numerical features."""
    numeric_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    _save_fig(fig, "05_correlation_heatmap.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Histograms of all numerical features in a grid."""
    features = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#26A69A", "#AB47BC", "#FFA726", "#42A5F5"]
    for ax, col, color in zip(axes.flat, features, colors):
        ax.hist(df[col], bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency")
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "06_feature_distributions.png")


def plot_monthly_trend(df: pd.DataFrame) -> None:
    """Average sales by month across all years."""
    df_copy = df.copy()
    df_copy["Month"] = df_copy["Date"].dt.month
    monthly = df_copy.groupby("Month")["Weekly_Sales"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly.index, monthly.values, color="#7E57C2", edgecolor="white")
    ax.set_title("Average Weekly Sales by Month", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    _save_fig(fig, "07_monthly_trend.png")


def generate_statistics_report(df: pd.DataFrame) -> None:
    """Save a text-based statistical summary to reports/metrics/."""
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "eda_summary.txt"

    lines = [
        "=" * 60,
        "WALMART SALES -- EDA SUMMARY",
        "=" * 60,
        "",
        f"Total Rows      : {len(df)}",
        f"Total Columns   : {df.shape[1]}",
        f"Stores          : {df['Store'].nunique()}",
        f"Date Range      : {df['Date'].min().date()} to {df['Date'].max().date()}",
        f"Weeks per Store : {df.groupby('Store')['Date'].count().mean():.0f} (avg)",
        "",
        "--- Target Variable (Weekly_Sales) ---",
        f"  Mean           : ${df['Weekly_Sales'].mean():>14,.2f}",
        f"  Median         : ${df['Weekly_Sales'].median():>14,.2f}",
        f"  Std Dev        : ${df['Weekly_Sales'].std():>14,.2f}",
        f"  Min            : ${df['Weekly_Sales'].min():>14,.2f}",
        f"  Max            : ${df['Weekly_Sales'].max():>14,.2f}",
        "",
        "--- Holiday Impact ---",
        f"  Non-Holiday Avg: ${df[df['Holiday_Flag']==0]['Weekly_Sales'].mean():>14,.2f}",
        f"  Holiday Avg    : ${df[df['Holiday_Flag']==1]['Weekly_Sales'].mean():>14,.2f}",
        "",
        "--- Missing Values ---",
        df.isnull().sum().to_string(),
        "",
        "--- Descriptive Statistics ---",
        df.describe().to_string(),
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("  Saved -> %s", path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Generate all EDA outputs."""
    logger.info("[1/8] Sales distribution...")
    plot_sales_distribution(df)

    logger.info("[2/8] Sales over time...")
    plot_sales_over_time(df)

    logger.info("[3/8] Sales by store...")
    plot_sales_by_store(df)

    logger.info("[4/8] Holiday impact...")
    plot_holiday_impact(df)

    logger.info("[5/8] Correlation heatmap...")
    plot_correlation_heatmap(df)

    logger.info("[6/8] Feature distributions...")
    plot_feature_distributions(df)

    logger.info("[7/8] Monthly trend...")
    plot_monthly_trend(df)

    logger.info("[8/8] Statistics report...")
    generate_statistics_report(df)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Phase 2b -- Exploratory Data Analysis")
    logger.info("=" * 50)

    # Run cleaning first (or load if already exists)
    interim_path = PROJECT_ROOT / PATHS["interim_data"]
    if interim_path.exists():
        logger.info("Loading cleaned data from %s", interim_path)
        df = load_parquet(str(interim_path))
    else:
        logger.info("Cleaned data not found -- running preprocessing first...")
        df = run_preprocessing()

    logger.info("")
    run_eda(df)
    logger.info("")
    logger.info("Phase 2 EDA complete! Check reports/figures/ and reports/metrics/")
