# Walmart Sales API

A FastAPI-based forecasting service for Walmart sales. The project loads a trained model and exposes prediction endpoints for single and batch requests, with health and drift report support.

## Requirements

- Python 3.11+
- Dependencies from `pyproject.toml`

## Install

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> If you prefer the `pyproject.toml` project format, use a modern toolchain such as `pip install .`.

## Run locally

Use the root launcher or the actual app module:

```bash
uvicorn main:app --reload
```

or

```bash
uvicorn api.main:app --reload
```

The service will start on `http://127.0.0.1:8000` by default.

## API Endpoints

- `GET /health` - Returns service status, model name, and target.
- `POST /predict` - Single prediction request.
- `POST /predict/batch` - Batch prediction request.
- `GET /drift` - Returns drift monitoring report.

### Prediction payload

Example request body for `/predict`:

```json
{
  "Store": 1,
  "Holiday_Flag": 0,
  "Temperature": 75.0,
  "Fuel_Price": 3.5,
  "CPI": 130.0,
  "Unemployment": 5.0,
  "Month": 1,
  "WeekOfYear": 1,
  "Year": 2022
}
```

For `/predict/batch`, send a JSON array of the same objects.

## Tests

Run tests with:

```bash
pytest
```

## Notes

- The API depends on model artifacts in `models/`, including `best_model.joblib` and `feature_columns.json`.
- Logs are written to `logs/predictions.jsonl` when predictions are made.
- The root `main.py` file exists to support FastAPI Cloud auto-discovery; the real app implementation is in `api/main.py`.
