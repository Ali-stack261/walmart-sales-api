import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint() -> None:
    payload = {
        "Store": 1,
        "Holiday_Flag": 0,
        "Temperature": 55.0,
        "Fuel_Price": 3.4,
        "CPI": 210.0,
        "Unemployment": 7.5,
        "Month": 8,
        "WeekOfYear": 32,
        "Year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_out_of_range_payload() -> None:
    payload = {
        "Store": 1,
        "Holiday_Flag": 0,
        "Temperature": 55.0,
        "Fuel_Price": 3.4,
        "CPI": -10.0,
        "Unemployment": 7.5,
        "Month": 8,
        "WeekOfYear": 32,
        "Year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_endpoint() -> None:
    payload = [
        {
            "Store": 1,
            "Holiday_Flag": 0,
            "Temperature": 55.0,
            "Fuel_Price": 3.4,
            "CPI": 210.0,
            "Unemployment": 7.5,
            "Month": 8,
            "WeekOfYear": 32,
            "Year": 2024,
        },
        {
            "Store": 2,
            "Holiday_Flag": 1,
            "Temperature": 60.0,
            "Fuel_Price": 3.6,
            "CPI": 215.0,
            "Unemployment": 8.1,
            "Month": 9,
            "WeekOfYear": 35,
            "Year": 2024,
        },
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 2


def test_prediction_logging_writes_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")
    payload = {
        "Store": 1,
        "Holiday_Flag": 0,
        "Temperature": 55.0,
        "Fuel_Price": 3.4,
        "CPI": 210.0,
        "Unemployment": 7.5,
        "Month": 8,
        "WeekOfYear": 32,
        "Year": 2024,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    log_file = tmp_path / "predictions.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["input"]["Store"] == 1
    assert "prediction" in entry
