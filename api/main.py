"""
FastAPI app for the flight delay prediction model.

Endpoints:
    GET  /health   - simple check that the API and model are up
    POST /predict  - takes one flight's features, returns delay probability

Run from project root:
    uvicorn api.main:app --reload
"""

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import FlightInput, PredictionOutput

MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

app = FastAPI(title="Flight Delay Predictor", version="1.0")

# Loaded once at startup, not per-request
model = None
category_maps = None


@app.on_event("startup")
def load_model():
    global model, category_maps
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        category_maps = pickle.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(flight: FlightInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = flight.model_dump()
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

    return PredictionOutput(
        delay_probability=prob,
        predicted_delayed=prob >= 0.5,
    )