"""
Preprocessing module -- Phase 2

Cleans the raw Walmart sales dataset:
  1. Parses Date column to datetime
  2. Sorts by Store + Date
  3. Removes duplicate rows
  4. Validates data types
  5. Saves cleaned output to data/interim/cleaned.parquet

Usage:
    python -m src.preprocessing
"""

import pandas as pd

from src.config import PATHS, PROJECT_ROOT
from src.logger import get_logger
from src.utils import load_parquet, save_parquet

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load the raw parquet dataset."""
    path = PROJECT_ROOT / PATHS["raw_data"]
    df = load_parquet(str(path))
    logger.info("Loaded raw data -- shape: %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw DataFrame.

    Steps:
        1. Parse Date to datetime
        2. Drop duplicate rows
        3. Sort by Store and Date
        4. Reset index
        5. Enforce correct dtypes
    """
    logger.info("Starting data cleaning...")
    initial_rows = len(df)

    # 1. Parse Date (format: DD-MM-YYYY)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    logger.info("  [1/5] Parsed Date column to datetime")

    # 2. Drop duplicates
    df = df.drop_duplicates()
    dropped = initial_rows - len(df)
    logger.info("  [2/5] Dropped %d duplicate rows", dropped)

    # 3. Sort by Store, then Date
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    logger.info("  [3/5] Sorted by Store + Date")

    # 4. Enforce dtypes
    df["Store"] = df["Store"].astype(int)
    df["Holiday_Flag"] = df["Holiday_Flag"].astype(int)
    logger.info("  [4/5] Enforced integer dtypes on Store, Holiday_Flag")

    # 5. Validate -- no nulls should exist
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        logger.warning("  [5/5] Found %d null values:\n%s", total_nulls, null_counts)
    else:
        logger.info("  [5/5] Validated: 0 null values")

    logger.info("Cleaning complete -- final shape: %s", df.shape)
    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Save the cleaned DataFrame to data/interim/cleaned.parquet."""
    path = str(PROJECT_ROOT / PATHS["interim_data"])
    save_parquet(df, path)
    logger.info("Saved cleaned data -> %s", path)


def run_preprocessing() -> pd.DataFrame:
    """Execute the full cleaning pipeline and return the cleaned DataFrame."""
    df = load_raw_data()
    df = clean_data(df)
    save_cleaned_data(df)
    return df


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Phase 2a -- Data Cleaning")
    logger.info("=" * 50)
    df = run_preprocessing()
    logger.info("")
    logger.info("Summary:")
    logger.info("  Rows    : %d", len(df))
    logger.info("  Columns : %d", df.shape[1])
    logger.info("  Date range: %s to %s", df["Date"].min(), df["Date"].max())
    logger.info("  Stores  : %d", df["Store"].nunique())
