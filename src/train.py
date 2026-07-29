"""
Model Training & Experiment Tracking -- Phase 4

Trains Random Forest, XGBoost, and LightGBM models.
Tracks experiments, hyperparameters, and metrics using MLflow.
Saves the best-performing model (based on Validation RMSE) as best_model.joblib.

Usage:
    python -m src.train
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import mlflow

from src.config import CONFIG, MLFLOW_TRACKING_URI, PATHS, PROJECT_ROOT, RANDOM_SEED, TARGET
from src.logger import get_logger
from src.utils import load_json, load_parquet, save_model

logger = get_logger(__name__)


def evaluate_model(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_training() -> None:
    # 1. Load data
    train_path = PROJECT_ROOT / PATHS["train_data"]
    val_path = PROJECT_ROOT / PATHS["validation_data"]
    
    if not train_path.exists() or not val_path.exists():
        logger.error("Train/Validation datasets not found. Run Phase 3 first.")
        return

    train_df = load_parquet(str(train_path))
    val_df = load_parquet(str(val_path))

    # 2. Load feature schema
    schema = load_json(str(PROJECT_ROOT / PATHS["feature_columns"]))
    features = schema["features"]

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_val = val_df[features]
    y_val = val_df[TARGET]

    logger.info("Training data: %d rows", len(X_train))
    logger.info("Validation data: %d rows", len(X_val))

    # 3. Setup MLflow
    # Using local SQLite tracking
    tracking_uri = str(PROJECT_ROOT / "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")
    mlflow.set_experiment("walmart_sales_forecasting")

    models_to_train = {
        "RandomForest": RandomForestRegressor(
            n_estimators=CONFIG["model"]["random_forest"]["n_estimators"],
            max_depth=CONFIG["model"]["random_forest"]["max_depth"],
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    best_model = None
    best_rmse = float("inf")
    best_model_name = ""

    # 4. Train and track
    for model_name, model in models_to_train.items():
        logger.info("\n--- Training %s ---", model_name)
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)

            # Predict
            val_preds = model.predict(X_val)

            # Evaluate
            metrics = evaluate_model(y_val, val_preds)
            logger.info("  Validation RMSE: $%.2f", metrics["rmse"])
            logger.info("  Validation MAE : $%.2f", metrics["mae"])
            logger.info("  Validation R2  : %.4f", metrics["r2"])

            # Log to MLflow
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Simple model tracking
            if model_name == "RandomForest":
                mlflow.sklearn.log_model(model, artifact_path="model")
            elif model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            elif model_name == "LightGBM":
                mlflow.lightgbm.log_model(model, artifact_path="model")

            # Check if best model
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = model
                best_model_name = model_name

    # 5. Save the best model globally for FastAPI inference
    logger.info("\n==================================================")
    logger.info("Training complete!")
    logger.info("Best model based on Val RMSE: %s (RMSE: $%.2f)", best_model_name, best_rmse)
    
    best_model_path = str(PROJECT_ROOT / PATHS["model"])
    save_model(best_model, best_model_path)
    logger.info("Saved best model -> %s", best_model_path)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Phase 4 -- Model Training & Experiment Tracking")
    logger.info("=" * 50)
    run_training()
