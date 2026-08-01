"""Simple drift monitoring utility for API inputs and training stats."""

import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.logger import get_logger
from src.utils import load_parquet

logger = get_logger(__name__)


def generate_drift_report(output_path: str | None = None) -> dict:
    """Create a lightweight drift summary comparing live inputs to the training dataset."""
    output_path = Path(output_path or str(PROJECT_ROOT / "reports" / "drift" / "drift_report.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training_data = load_parquet(str(PROJECT_ROOT / "data" / "processed" / "train.parquet"))
    if training_data.empty:
        raise ValueError("Training data is not available for drift analysis")

    reference = training_data[["Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Month", "WeekOfYear", "Year"]]
    summary = {
        "rows": int(len(reference)),
        "columns": int(len(reference.columns)),
        "store_range": [int(reference["Store"].min()), int(reference["Store"].max())],
        "temperature_range": [float(reference["Temperature"].min()), float(reference["Temperature"].max())],
        "fuel_price_range": [float(reference["Fuel_Price"].min()), float(reference["Fuel_Price"].max())],
        "cpi_range": [float(reference["CPI"].min()), float(reference["CPI"].max())],
        "unemployment_range": [float(reference["Unemployment"].min()), float(reference["Unemployment"].max())],
        "year_range": [int(reference["Year"].min()), int(reference["Year"].max())],
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote drift report -> %s", output_path)
    return summary


if __name__ == "__main__":
    generate_drift_report()
