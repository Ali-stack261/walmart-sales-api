"""
Shared utility functions used across the project.
"""

import json
from pathlib import Path

import joblib
import pandas as pd


def save_model(model, path: str) -> None:
    """Serialize a model to disk using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load a serialized model from disk."""
    return joblib.load(path)


def save_json(data: dict, path: str) -> None:
    """Save a dictionary as a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load a JSON file and return a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to Parquet format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame."""
    return pd.read_parquet(path)
