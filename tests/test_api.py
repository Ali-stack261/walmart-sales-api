from fastapi.testclient import TestClient

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
