"""
Configuration loader.

Reads config/config.yaml and .env so every other module
can import settings from a single place.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Load .env into os.environ ──
load_dotenv()

# ── Project root (two levels up from src/config.py) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load YAML config ──
_config_path = PROJECT_ROOT / "config" / "config.yaml"
with open(_config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# ── Convenience accessors ──
RANDOM_SEED: int = CONFIG["random_seed"]
TEST_SIZE: float = CONFIG["test_size"]
VALIDATION_SIZE: float = CONFIG["validation_size"]
TARGET: str = CONFIG["target"]
PATHS: dict = CONFIG["paths"]

# ── Environment variables ──
MODEL_PATH = os.getenv("MODEL_PATH", PATHS["model"])
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
APP_NAME = os.getenv("APP_NAME", "Walmart Sales API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
