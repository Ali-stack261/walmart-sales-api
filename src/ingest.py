"""
Data Ingestion Script — Phase 1

Copies the raw Walmart sales dataset into the project's data/raw/ directory
and performs a quick sanity check to ensure the file is valid.

Usage:
    python src/ingest.py
"""

import shutil
from pathlib import Path

import pandas as pd

from src.config import PATHS, PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)


def ingest_data() -> pd.DataFrame:
    """
    Copy the raw dataset into data/raw/ and return the loaded DataFrame.

    If the file already exists in data/raw/, it is loaded directly.
    Otherwise, it is copied from the project root.
    """
    raw_dest = PROJECT_ROOT / PATHS["raw_data"]
    raw_dest.parent.mkdir(parents=True, exist_ok=True)

    # Source file sitting at the project root
    source_file = PROJECT_ROOT / "walmart_sales.parquet"

    if not raw_dest.exists():
        if not source_file.exists():
            raise FileNotFoundError(
                f"Source dataset not found at {source_file}. "
                "Please place walmart_sales.parquet in the project root."
            )
        shutil.copy2(source_file, raw_dest)
        logger.info("Copied raw dataset -> %s", raw_dest)
    else:
        logger.info("Raw dataset already exists at %s", raw_dest)

    # -- Quick sanity check --
    df = pd.read_parquet(raw_dest)
    logger.info("Dataset loaded - shape: %s", df.shape)
    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    logger.info("Missing values:\n%s", df.isnull().sum().to_string())

    return df


def create_project_directories() -> None:
    """Create all required project directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "models",
        "notebooks",
        "reports/figures",
        "reports/metrics",
        "reports/drift",
        "docs",
        "logs",
    ]
    for d in dirs:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info("[OK] %s", path)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Phase 1 -- Environment Setup & Data Ingestion")
    logger.info("=" * 50)

    # Step 1: Create directory structure
    logger.info("")
    logger.info("[1/2] Creating project directories...")
    create_project_directories()

    # Step 2: Ingest raw data
    logger.info("")
    logger.info("[2/2] Ingesting raw dataset...")
    df = ingest_data()

    # Step 3: Summary
    logger.info("")
    logger.info("Phase 1 complete!")
    logger.info("   Rows       : %d", len(df))
    logger.info("   Columns    : %d", df.shape[1])
    logger.info("   Target     : Weekly_Sales")
    logger.info("   Raw path   : %s", PATHS["raw_data"])
