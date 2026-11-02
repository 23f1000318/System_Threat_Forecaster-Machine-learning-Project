"""Streamlit interface for System Threat Forecaster."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from train import prepare_frame


MODEL_PATH = Path("model/model.joblib")


@st.cache_resource
def load_bundle():
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError("model/model.joblib is missing from the repository.")
    return joblib.load(MODEL_PATH)


def predict_records(raw: pd.DataFrame):
    bundle = load_bundle()
    features, _ = prepare_frame(
        raw,
        training=False,
        retained_columns=bundle["retained_columns"],
    )
    features = features.reindex(columns=bundle["model_input_columns"])
    transformed = bundle["preprocessor"].transform(features)
    selected = bundle["selector"].transform(transformed)
    model = bundle["model"]
    return model.predict(selected).astype(int), model.predict_proba(selected)[:, 1]


st.set_page_config(page_title="System Threat Forecaster", page_icon="🛡️")
st.title("🛡️ System Threat Forecaster")
st.write(
    "Predict whether a Windows machine is likely to have malware detected from "
    "antivirus and system telemetry."
)

try:
    load_bundle()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

tab_json, tab_csv = st.tabs(["Single machine (JSON)", "Batch prediction (CSV)"])

with tab_json:
    default_json = json.dumps(
        {
            "ProductName": "Windows Defender",
            "HasTpm": 1,
            "NumAntivirusProductsInstalled": 1,
        },
        indent=2,
    )
    text = st.text_area("Machine telemetry as JSON", value=default_json, height=220)
    if st.button("Predict", key="json_predict"):
        try:
            record = json.loads(text)
            if not isinstance(record, dict):
                raise ValueError("JSON must describe one machine as an object.")
            predictions, probabilities = predict_records(pd.DataFrame([record]))
            st.metric("Prediction", int(predictions[0]))
            st.metric("Malware probability", f"{probabilities[0]:.2%}")
        except Exception as exc:
            st.error(str(exc))

with tab_csv:
    uploaded = st.file_uploader("Upload a CSV containing machine records", type=["csv"])
    if uploaded is not None and st.button("Predict batch", key="csv_predict"):
        try:
            frame = pd.read_csv(uploaded)
            predictions, probabilities = predict_records(frame)
            result = frame.copy()
            result["prediction"] = predictions
            result["malware_probability"] = probabilities.round(6)
            st.dataframe(result[["prediction", "malware_probability"]])
            st.download_button(
                "Download predictions",
                result[["prediction", "malware_probability"]].to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(str(exc))
