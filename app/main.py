"""Serwowanie modelu predykcji anulowań — FastAPI + Prometheus.

Endpointy:
- GET  /health   — status serwisu i załadowanego modelu
- POST /predict  — predykcja dla pojedynczej rezerwacji (JSON)
- GET  /metrics  — metryki Prometheus (liczniki predykcji, latencja, błędy)

Każda predykcja jest dopisywana do logu JSONL (PREDICTIONS_LOG); na tej
podstawie scripts/check_drift.py wykonuje prosty test dryfu danych.
Ścieżki artefaktów są konfigurowalne przez zmienne środowiskowe, dzięki czemu
ten sam kod działa lokalnie i w kontenerze (model montowany jako wolumen).
"""

import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from app.preprocessing import prepare_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "data/06_models/model.pkl"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "data/06_models/model_features.json"))
PREDICTIONS_LOG = Path(os.getenv("PREDICTIONS_LOG", "logs/predictions.jsonl"))

# Próg decyzyjny klasyfikacji (P(anulowana) >= próg -> klasa 1).
# Wartość 0.5 to standard; można ją dostroić pod koszty biznesowe hotelu.
DECISION_THRESHOLD = 0.5

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Liczba predykcji wg przewidzianej klasy", ["predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Czas obsługi pojedynczej predykcji"
)
PREDICTION_ERRORS = Counter("prediction_errors_total", "Liczba błędów predykcji")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    with open(MODEL_PATH, "rb") as f:
        state["model"] = pickle.load(f)
    with open(FEATURES_PATH, encoding="utf-8") as f:
        state["features"] = json.load(f)
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    yield
    state.clear()


app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="Predykcja anulowania rezerwacji hotelowej (projekt ASI).",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


class Booking(BaseModel):
    """Surowe pola rezerwacji — jak w danych wejściowych (bez kolumn-wycieku)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
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
            ]
        }
    )

    hotel: str
    lead_time: int = Field(ge=0)
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float = 0
    babies: int = 0
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int = 0
    deposit_type: str
    agent: float | None = None
    days_in_waiting_list: int = 0
    customer_type: str
    adr: float
    required_car_parking_spaces: int = 0
    total_of_special_requests: int = 0


class Prediction(BaseModel):
    prediction: int
    probability: float
    label: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "n_features": len(state.get("features", [])),
    }


@app.post("/predict", response_model=Prediction)
def predict(booking: Booking) -> Prediction:
    start = time.perf_counter()
    try:
        features = prepare_features(booking.model_dump(), state["features"])
        probability = float(state["model"].predict_proba(features)[0, 1])
    except Exception as exc:  # nieprzewidziane dane wejściowe
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = int(probability >= DECISION_THRESHOLD)
    PREDICTION_LATENCY.observe(time.perf_counter() - start)
    PREDICTIONS_TOTAL.labels(predicted_class=str(prediction)).inc()
    _log_prediction(booking, prediction, probability)

    return Prediction(
        prediction=prediction,
        probability=round(probability, 4),
        label="anulowana" if prediction == 1 else "zrealizowana",
    )


def _log_prediction(booking: Booking, prediction: int, probability: float) -> None:
    """Dopisuje predykcję do logu JSONL (wejście do testu dryfu)."""
    record = {
        "ts": time.time(),
        "lead_time": booking.lead_time,
        "adr": booking.adr,
        "total_nights": booking.stays_in_weekend_nights + booking.stays_in_week_nights,
        "country": booking.country,
        "prediction": prediction,
        "probability": round(probability, 4),
    }
    with PREDICTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
