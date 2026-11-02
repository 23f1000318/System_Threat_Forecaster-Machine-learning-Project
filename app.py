"""FastAPI service for System Threat Forecaster predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from train import prepare_frame


MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/model.joblib"))
app = FastAPI(title="System Threat Forecaster API", version="1.0.0")
bundle = None


def get_bundle():
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run train.py before starting the API."
            )
        bundle = joblib.load(MODEL_PATH)
    return bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(payload: dict[str, Any]):
    if not payload:
        raise HTTPException(status_code=400, detail="Request body must contain machine features.")
    try:
        model_bundle = get_bundle()
        raw = pd.DataFrame([payload])
        features, _ = prepare_frame(
            raw,
            training=False,
            retained_columns=model_bundle["retained_columns"],
        )
        features = features.reindex(columns=model_bundle["model_input_columns"])
        transformed = model_bundle["preprocessor"].transform(features)
        selected = model_bundle["selector"].transform(transformed)
        model = model_bundle["model"]
        prediction = int(model.predict(selected)[0])
        probability = float(model.predict_proba(selected)[0, 1])
        return {
            "prediction": prediction,
            "malware_probability": round(probability, 6),
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
