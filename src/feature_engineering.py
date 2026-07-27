"""
Feature Engineering Module -- Phase 3

Extracts features, scales data, and splits the dataset into
train, validation, and test sets for model training.

Outputs:
  - data/processed/features.parquet (All engineered features before split)
  - data/processed/train.parquet
  - data/processed/validation.parquet
  - data/processed/test.parquet
  - models/scaler.joblib
  - models/feature_columns.json

Usage:
    python -m src.feature_engineering
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, PROJECT_ROOT, RANDOM_SEED, TARGET, TEST_SIZE, VALIDATION_SIZE
from src.logger import get_logger
from src.utils import load_parquet, save_json, save_model, save_parquet
from src.preprocessing import run_preprocessing

logger = get_logger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the Date column."""
    logger.info("Extracting time-based features from Date...")
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"] = df["Date"].dt.year
    # We can drop the original Date column since we have temporal features now
    df = df.drop(columns=["Date"])
    return df


def split_data(df: pd.DataFrame):
    """Split into train, validation, and test sets."""
    logger.info("Splitting dataset...")

    # First split: Train vs Temp (Validation + Test)
    train_df, temp_df = train_test_split(
        df, test_size=(TEST_SIZE + VALIDATION_SIZE), random_state=RANDOM_SEED
    )

    # Second split: Validation vs Test
    val_ratio = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio), random_state=RANDOM_SEED
    )

    logger.info("  Train      : %d rows", len(train_df))
    logger.info("  Validation : %d rows", len(val_df))
    logger.info("  Test       : %d rows", len(test_df))

    return train_df, val_df, test_df


def scale_features(train_df, val_df, test_df, num_cols):
    """Scale numerical columns using StandardScaler fit only on the training set."""
    logger.info("Scaling numerical features: %s", num_cols)
    scaler = StandardScaler()

    # Fit on train, transform all
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_df[num_cols] = scaler.transform(val_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    # Save scaler for inference later
    scaler_path = str(PROJECT_ROOT / PATHS["scaler"])
    save_model(scaler, scaler_path)
    logger.info("  Saved scaler -> %s", scaler_path)

    return train_df, val_df, test_df


def run_feature_engineering() -> None:
    """Run the complete feature engineering and splitting pipeline."""
    # 1. Load cleaned data
    interim_path = PROJECT_ROOT / PATHS["interim_data"]
    if not interim_path.exists():
        logger.info("Cleaned data not found. Running preprocessing...")
        df = run_preprocessing()
    else:
        df = load_parquet(str(interim_path))

    # 2. Extract features
    df = create_time_features(df)

    # Define feature categories
    categorical_cols = ["Store", "Holiday_Flag", "Month", "WeekOfYear", "Year"]
    numerical_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    features = categorical_cols + numerical_cols

    # Save complete feature dataset (unscaled, unsplit) for reference
    save_parquet(df, str(PROJECT_ROOT / PATHS["processed_data"]))
    
    # Save feature schema
    schema = {
        "features": features,
        "categorical": categorical_cols,
        "numerical": numerical_cols,
        "target": TARGET,
    }
    save_json(schema, str(PROJECT_ROOT / PATHS["feature_columns"]))
    logger.info("Saved feature schema -> %s", PATHS["feature_columns"])

    # 3. Split data
    train_df, val_df, test_df = split_data(df)

    # 4. Scale features
    train_df, val_df, test_df = scale_features(train_df, val_df, test_df, numerical_cols)

    # 5. Save splits
    save_parquet(train_df, str(PROJECT_ROOT / PATHS["train_data"]))
    save_parquet(val_df, str(PROJECT_ROOT / PATHS["validation_data"]))
    save_parquet(test_df, str(PROJECT_ROOT / PATHS["test_data"]))
    
    logger.info("Saved train, val, and test datasets to data/processed/")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Phase 3 -- Feature Engineering")
    logger.info("=" * 50)
    run_feature_engineering()
    logger.info("")
    logger.info("Phase 3 complete! Data is ready for Model Training.")
