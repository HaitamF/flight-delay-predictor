"""
Basic tests for the flight delay prediction API.

Uses FastAPI's TestClient, which calls the app directly in-process —
no need for uvicorn to be running separately.

Run from project root:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
import pytest
from api.main import app

VALID_FLIGHT = {
    "MONTH": 7,
    "DAY_OF_WEEK": 5,
    "DEP_TIME_BLK": "1700-1759",
    "DISTANCE_GROUP": 4,
    "CARRIER_NAME": "Delta Air Lines Inc.",
    "DEPARTING_AIRPORT": "Atlanta Municipal",
}


@pytest.fixture
def client():
    # Using TestClient as a context manager triggers startup/shutdown
    # events (like loading the model) — plain instantiation skips them.
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_flight(client):
    response = client.post("/predict", json=VALID_FLIGHT)
    assert response.status_code == 200
    body = response.json()
    assert "delay_probability" in body
    assert "predicted_delayed" in body
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert isinstance(body["predicted_delayed"], bool)


def test_predict_unknown_carrier_rejected(client):
    bad_flight = VALID_FLIGHT.copy()
    bad_flight["CARRIER_NAME"] = "Definitely Not A Real Airline"
    response = client.post("/predict", json=bad_flight)
    assert response.status_code == 400


def test_predict_unknown_airport_rejected(client):
    bad_flight = VALID_FLIGHT.copy()
    bad_flight["DEPARTING_AIRPORT"] = "Made Up Airport"
    response = client.post("/predict", json=bad_flight)
    assert response.status_code == 400


def test_predict_missing_field(client):
    incomplete_flight = VALID_FLIGHT.copy()
    del incomplete_flight["MONTH"]
    response = client.post("/predict", json=incomplete_flight)
    assert response.status_code == 422  # Pydantic validation error