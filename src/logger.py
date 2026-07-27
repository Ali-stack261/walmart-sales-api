"""
Logger setup.

Reads config/logging.yaml so every module gets consistent logging
via ``from src.logger import get_logger``.
"""

import logging
import logging.config
from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

# Ensure log directory exists
_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)

# Load logging configuration
_logging_config_path = PROJECT_ROOT / "config" / "logging.yaml"
with open(_logging_config_path, "r") as f:
    _log_cfg = yaml.safe_load(f)

logging.config.dictConfig(_log_cfg)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)
