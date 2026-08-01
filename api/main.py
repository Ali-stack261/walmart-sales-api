from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PATHS, PROJECT_ROOT, TARGET
from src.logger import get_logger
from src.utils import load_json, load_model

logger = get_logger(__name__)

app = FastAPI(title="Walmart Sales Forecasting API", version="1.0.0")


class PredictionRequest(BaseModel):
    Store: int = Field(..., ge=1)
    Holiday_Flag: int = Field(..., ge=0, le=1)
    Temperature: float = Field(...)
    Fuel_Price: float = Field(...)
    CPI: float = Field(...)
    Unemployment: float = Field(...)
    Month: int = Field(..., ge=1, le=12)
    WeekOfYear: int = Field(..., ge=1, le=53)
    Year: int = Field(..., ge=2010)


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model: str


MODEL_PATH = PROJECT_ROOT / PATHS["model"]
FEATURE_SCHEMA_PATH = PROJECT_ROOT / PATHS["feature_columns"]


def _load_model_and_features():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    if not FEATURE_SCHEMA_PATH.exists():
        raise RuntimeError(f"Feature schema not found at {FEATURE_SCHEMA_PATH}")

    model = load_model(str(MODEL_PATH))
    schema = load_json(str(FEATURE_SCHEMA_PATH))
    return model, schema["features"]


MODEL, FEATURE_COLUMNS = _load_model_and_features()


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
        import pandas as pd

        df = pd.DataFrame([feature_row])
        df = df[FEATURE_COLUMNS]
        prediction = float(MODEL.predict(df)[0])
        return PredictionResponse(prediction=prediction, target=TARGET, model="best_model")
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
