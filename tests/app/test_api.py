"""Testy API serwującego model (FastAPI TestClient).

Wymagają artefaktów modelu (data/06_models/) — jeśli ich nie ma (np. w CI
bez danych), testy są pomijane. Lokalnie: najpierw `kedro run`.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODEL_PATH = Path("data/06_models/model.pkl")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="brak artefaktu modelu — uruchom najpierw: kedro run",
)

SAMPLE_BOOKING = {
    "hotel": "City Hotel",
    "lead_time": 120,
    "arrival_date_year": 2017,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 15,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "assigned_room_type": "A",
    "booking_changes": 0,
    "deposit_type": "No Deposit",
    "agent": 9,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 105.5,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
}


def test_health_zwraca_status_i_model():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["n_features"] > 0


def test_predict_zwraca_poprawna_predykcje():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_BOOKING)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("anulowana", "zrealizowana")


def test_predict_nieznany_kraj_idzie_do_other():
    booking = dict(SAMPLE_BOOKING, country="XYZ")  # kraj spoza treningu
    with TestClient(app) as client:
        response = client.post("/predict", json=booking)
    assert response.status_code == 200


def test_metrics_eksponuje_liczniki():
    with TestClient(app) as client:
        client.post("/predict", json=SAMPLE_BOOKING)
        response = client.get("/metrics/")
    assert response.status_code == 200
    assert "predictions_total" in response.text
