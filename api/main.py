import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PATHS, PROJECT_ROOT, TARGET
from src.drift_monitor import generate_drift_report
from src.logger import get_logger
from src.utils import load_json, load_model

logger = get_logger(__name__)

app = FastAPI(title="Walmart Sales Forecasting API", version="1.0.0")


class PredictionRequest(BaseModel):
    Store: int = Field(..., ge=1, le=45)
    Holiday_Flag: int = Field(..., ge=0, le=1)
    Temperature: float = Field(..., ge=-10, le=100)
    Fuel_Price: float = Field(..., ge=2.0, le=5.0)
    CPI: float = Field(..., ge=120.0, le=230.0)
    Unemployment: float = Field(..., ge=3.0, le=15.0)
    Month: int = Field(..., ge=1, le=12)
    WeekOfYear: int = Field(..., ge=1, le=53)
    Year: int = Field(..., ge=2010, le=2030)


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest]


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


MODEL_PATH = PROJECT_ROOT / PATHS["model"]
FEATURE_SCHEMA_PATH = PROJECT_ROOT / PATHS["feature_columns"]
PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"


def _load_model_and_features():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not FEATURE_SCHEMA_PATH.exists():
        raise RuntimeError(f"Feature schema not found at {FEATURE_SCHEMA_PATH}")

    model = load_model(str(MODEL_PATH))
    schema = load_json(str(FEATURE_SCHEMA_PATH))
    return model, schema["features"]


MODEL, FEATURE_COLUMNS = _load_model_and_features()


def _append_prediction_log(payload: dict, prediction: float) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": payload,
        "prediction": prediction,
    }
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": str(MODEL_PATH.name), "target": TARGET}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        feature_row = {
            "Store": payload.Store,
            "Holiday_Flag": payload.Holiday_Flag,
            "Temperature": payload.Temperature,
            "Fuel_Price": payload.Fuel_Price,
            "CPI": payload.CPI,
            "Unemployment": payload.Unemployment,
            "Month": payload.Month,
            "WeekOfYear": payload.WeekOfYear,
            "Year": payload.Year,
        }
        df = pd.DataFrame([feature_row])
        df = df[FEATURE_COLUMNS]
        prediction = float(MODEL.predict(df)[0])
        response = PredictionResponse(prediction=prediction, target=TARGET, model="best_model")
        _append_prediction_log(feature_row, response.prediction)
        return response
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: list[PredictionRequest]) -> BatchPredictionResponse:
    try:
        predictions = []
        for request in payload:
            feature_row = {
                "Store": request.Store,
                "Holiday_Flag": request.Holiday_Flag,
                "Temperature": request.Temperature,
                "Fuel_Price": request.Fuel_Price,
                "CPI": request.CPI,
                "Unemployment": request.Unemployment,
                "Month": request.Month,
                "WeekOfYear": request.WeekOfYear,
                "Year": request.Year,
            }
            df = pd.DataFrame([feature_row])
            df = df[FEATURE_COLUMNS]
            prediction = float(MODEL.predict(df)[0])
            response = PredictionResponse(prediction=prediction, target=TARGET, model="best_model")
            predictions.append(response)
            _append_prediction_log(feature_row, response.prediction)
        return BatchPredictionResponse(predictions=predictions)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/drift")
def drift() -> dict:
    return generate_drift_report()
