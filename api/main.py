"""
FastAPI app for the flight delay prediction model.

Endpoints:
    GET  /health   - simple check that the API and model are up
    POST /predict  - takes one flight's features, returns delay probability

Run from project root:
    uvicorn api.main:app --reload
"""

import pickle
import json
import os
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import FlightInput, PredictionOutput

MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

app = FastAPI(title="Flight Delay Predictor", version="1.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for a portfolio demo; restrict in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request
model = None
category_maps = None
airport_coords = None
priors = {}  # carrier_avg_delay, airport_avg_delay, day_avg_delay, timeblock_avg_delay


@app.on_event("startup")
def load_model():
    global model, category_maps, airport_coords, priors
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        category_maps = pickle.load(f)
    with open("api/airport_coordinates.json", "r") as f:
        airport_coords = json.load(f)

    for prior_name, filename in [
        ("carrier_avg_delay", "api/carrier_avg_delay.json"),
        ("airport_avg_delay", "api/airport_avg_delay.json"),
        ("day_avg_delay", "api/day_avg_delay.json"),
        ("timeblock_avg_delay", "api/timeblock_avg_delay.json"),
    ]:
        with open(filename, "r") as f:
            priors[prior_name] = json.load(f)


def _get_weather(airport: str) -> dict:
    """Fetch current weather for an airport via Open-Meteo (free, no key)."""
    coords = airport_coords.get(airport)
    if coords is None:
        raise HTTPException(
            status_code=400,
            detail=f"No coordinates on file for airport '{airport}'."
        )

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "precipitation,snowfall,temperature_2m,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        current = resp.json()["current"]
    except Exception:
        # Weather API unreachable — fall back to neutral defaults rather than failing the whole request
        return {"PRCP": 0.0, "SNOW": 0.0, "SNWD": 0.0, "TMAX": 70.0, "AWND": 5.0}

    return {
        "PRCP": current.get("precipitation", 0.0),
        "SNOW": current.get("snowfall", 0.0),
        "SNWD": 0.0,  # Open-Meteo's free tier doesn't expose snow depth directly
        "TMAX": current.get("temperature_2m", 70.0),
        "AWND": current.get("wind_speed_10m", 5.0),
    }


def _get_priors(carrier: str, airport: str, day_of_week: int, dep_time_blk: str) -> dict:
    """Look up historical average delay rates computed during feature engineering."""
    def lookup(table, key):
        val = table.get(str(key))
        if val is None:
            raise HTTPException(status_code=400, detail=f"No historical prior found for '{key}'.")
        return val

    return {
        "carrier_avg_delay": lookup(priors["carrier_avg_delay"], carrier),
        "airport_avg_delay": lookup(priors["airport_avg_delay"], airport),
        "day_avg_delay": lookup(priors["day_avg_delay"], day_of_week),
        "timeblock_avg_delay": lookup(priors["timeblock_avg_delay"], dep_time_blk),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(flight: FlightInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = flight.model_dump()

    # Enrich: fetch live weather + look up historical priors,
    # so the user never has to know these values themselves.
    weather = _get_weather(row["DEPARTING_AIRPORT"])
    hist_priors = _get_priors(
        row["CARRIER_NAME"], row["DEPARTING_AIRPORT"],
        row["DAY_OF_WEEK"], row["DEP_TIME_BLK"]
    )
    row.update(weather)
    row.update(hist_priors)

    df = pd.DataFrame([row])

    # Apply the same categorical -> code mapping used at training time
    for map_name, mapping in category_maps.items():
        col = map_name.replace("_map", "").upper()
        if col in df.columns:
            code = mapping.get(df.at[0, col])
            if code is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown value '{df.at[0, col]}' for field '{col}' "
                           f"— not seen during training."
                )
            df[col] = code

    prob = float(model.predict_proba(df)[:, 1][0])
    predicted_delayed = prob >= 0.5

    _record_prediction(predicted_delayed)

    return PredictionOutput(
        delay_probability=prob,
        predicted_delayed=predicted_delayed,
    )


STATS_PATH = "api/prediction_stats.json"


def _record_prediction(predicted_delayed: bool):
    """Append-only counter of predictions made through this API, for the live dashboard."""
    stats = _load_stats()
    stats["total"] += 1
    if predicted_delayed:
        stats["predicted_delayed"] += 1
    else:
        stats["predicted_on_time"] += 1
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f)


def _load_stats():
    if not os.path.exists(STATS_PATH):
        return {"total": 0, "predicted_delayed": 0, "predicted_on_time": 0}
    with open(STATS_PATH, "r") as f:
        return json.load(f)


@app.get("/model-stats")
def get_model_stats():
    """Static model performance stats (from training/evaluation), for the dashboard."""
    with open("api/model_stats.json", "r") as f:
        return json.load(f)


@app.get("/stats")
def get_stats():
    """Live usage stats: how many predictions made through this API, and their split."""
    return _load_stats()