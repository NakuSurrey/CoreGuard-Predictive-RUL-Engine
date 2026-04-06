"""
FastAPI Application — CoreGuard Predictive RUL Engine.

This is the web server. It listens for HTTP requests, processes sensor data,
and returns RUL predictions as JSON.

How the API works (request flow):

    [Client sends POST /predict with sensor JSON]
        ↓
    [FastAPI validates the JSON against SensorInput schema]
        ↓
    [If invalid → returns 422 error with details]
    [If valid → continues]
        ↓
    [Convert sensor values into a DataFrame row]
        ↓
    [Normalize the row using the training scaler]
        ↓
    [Add rolling and lag features (set to 0 for single-row prediction)]
        ↓
    [Feed the row to XGBoost model]
        ↓
    [Model returns predicted RUL]
        ↓
    [Return JSON response to client]

For /predict/explain, there is one extra step after prediction:
    [Feed the row to SHAP explainer]
        ↓
    [SHAP returns each feature's contribution]
        ↓
    [Return JSON response with prediction + explanation]

Three endpoints:
    GET  /health           — is the API running?
    POST /predict          — get predicted RUL (fast, no explanation)
    POST /predict/explain  — get predicted RUL + SHAP breakdown
"""

import sys
import os

# adding project root to path so imports work when running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    XGBOOST_MODEL_PATH,
    SCALER_PATH,
    USEFUL_SENSORS,
    USEFUL_SETTINGS,
)
from src.api.schemas import (
    SensorInput,
    PredictionResponse,
    ExplanationResponse,
    FeatureContribution,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# APP SETUP
# ---------------------------------------------------------------------------
# creating the FastAPI app — this is the object that uvicorn runs
app = FastAPI(
    title="CoreGuard Predictive RUL Engine",
    description=(
        "Predicts Remaining Useful Life (RUL) of industrial equipment "
        "using sensor data. Returns predictions and SHAP-based explanations."
    ),
    version="1.0.0",
)

# CORS middleware — allows the Streamlit dashboard (running on a different port)
# to make requests to this API. Without this, the browser would block requests
# from localhost:8501 (Streamlit) to localhost:8000 (FastAPI).
#
# allow_origins=["*"] means any domain can call this API.
# In production, this would be restricted to specific domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# MODEL LOADING (happens once when the server starts)
# ---------------------------------------------------------------------------
# storing the model and scaler in global variables so they stay in memory
# between requests. Loading them once at startup is much faster than
# loading from disk on every single request.
xgb_model = None
scaler = None


@app.on_event("startup")
def load_models():
    """
    Load the trained model and scaler when the API server starts.

    This runs ONCE when uvicorn starts the server.
    After this, xgb_model and scaler are in memory and ready for predictions.
    """
    global xgb_model, scaler

    # load the trained XGBoost model
    if XGBOOST_MODEL_PATH.exists():
        xgb_model = joblib.load(XGBOOST_MODEL_PATH)
        print(f"[api] XGBoost model loaded from {XGBOOST_MODEL_PATH}")
    else:
        print(f"[api] WARNING: Model file not found at {XGBOOST_MODEL_PATH}")
        print("[api] Run 'python scripts/train.py' first to train the model.")

    # load the fitted scaler (same one used during training)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        print(f"[api] Scaler loaded from {SCALER_PATH}")
    else:
        print(f"[api] WARNING: Scaler file not found at {SCALER_PATH}")


# ---------------------------------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------------------------------
def prepare_input(sensor_data: SensorInput) -> pd.DataFrame:
    """
    Convert the incoming sensor JSON into a DataFrame the model can read.

    Steps:
        1. Extract sensor values from the request into a dictionary
        2. Create a single-row DataFrame
        3. Normalize using the training scaler
        4. Add rolling and lag feature columns (set to 0 for single-row input)

    Why rolling and lag features are 0:
        Rolling averages need multiple past cycles to compute.
        Lag features need the previous cycle's value.
        For a single-row prediction, there is no history.
        Setting them to 0 is safe — the model still uses the raw sensor
        values and settings, which carry the most signal.

    Args:
        sensor_data: Validated SensorInput from the request.

    Returns:
        DataFrame with all 44 feature columns, ready for model.predict().
    """
    # step 1: build a dict of setting and sensor values
    raw_values = {}
    for col in USEFUL_SETTINGS:
        raw_values[col] = getattr(sensor_data, col)
    for col in USEFUL_SENSORS:
        raw_values[col] = getattr(sensor_data, col)

    # step 2: single-row DataFrame
    df = pd.DataFrame([raw_values])

    # step 3: normalize using the training scaler
    feature_cols = USEFUL_SETTINGS + USEFUL_SENSORS
    df[feature_cols] = scaler.transform(df[feature_cols])

    # step 4: add rolling feature columns (all rolling first, then all diff)
    # IMPORTANT: must match the exact column order from training.
    # In preprocessor.py, add_rolling_features() runs first (adds all rolling_*),
    # then add_lag_features() runs second (adds all diff_*).
    # If the order is wrong, XGBoost rejects the input.
    for sensor in USEFUL_SENSORS:
        df[f"rolling_{sensor}"] = df[sensor].values

    # step 5: add lag feature columns (all zeros for single-row prediction)
    for sensor in USEFUL_SENSORS:
        df[f"diff_{sensor}"] = 0.0

    return df


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.

    Returns whether the API is running and the model is loaded.
    Any monitoring system can ping this to check if the service is alive.
    """
    return HealthResponse(
        status="healthy" if xgb_model is not None else "unhealthy",
        model_loaded=xgb_model is not None,
        model_type="XGBoost" if xgb_model is not None else "none",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(sensor_data: SensorInput):
    """
    Predict Remaining Useful Life from sensor data.

    This is the FAST endpoint — prediction only, no SHAP explanation.

    Request flow:
        Client sends sensor JSON → validate → normalize → predict → return RUL

    Returns:
        JSON with predicted_rul, model_used, and status.
    """
    # check that model is loaded
    if xgb_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python scripts/train.py' first.",
        )

    # prepare the input DataFrame
    df = prepare_input(sensor_data)

    # get feature columns (same order as training)
    feature_cols = [col for col in df.columns if col not in {"unit_id", "cycle", "rul"}]
    X = df[feature_cols]

    # make prediction
    prediction = float(xgb_model.predict(X)[0])

    # RUL cannot be negative
    prediction = max(prediction, 0.0)

    return PredictionResponse(
        predicted_rul=round(prediction, 2),
        model_used="XGBoost",
        status="success",
    )


@app.post("/predict/explain", response_model=ExplanationResponse)
def predict_with_explanation(sensor_data: SensorInput):
    """
    Predict RUL AND return a SHAP-based explanation.

    This is the DETAILED endpoint — prediction + which sensors drove it.

    Request flow:
        Client sends sensor JSON → validate → normalize → predict →
        compute SHAP values → return RUL + feature contributions

    Returns:
        JSON with predicted_rul, base_value, model_used, feature_contributions, and status.
    """
    # check that model is loaded
    if xgb_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python scripts/train.py' first.",
        )

    # prepare the input DataFrame
    df = prepare_input(sensor_data)

    # get feature columns
    feature_cols = [col for col in df.columns if col not in {"unit_id", "cycle", "rul"}]

    # use the SHAP explainer from Phase 3
    from src.explainability.shap_explainer import explain_single_prediction

    explanation = explain_single_prediction(
        model=xgb_model,
        feature_values=df,
        feature_cols=feature_cols,
    )

    # convert the raw contribution dicts to FeatureContribution objects
    contributions = [
        FeatureContribution(
            feature=c["feature"],
            value=round(c["value"], 6),
            shap_value=round(c["shap_value"], 4),
        )
        for c in explanation["feature_contributions"]
    ]

    return ExplanationResponse(
        predicted_rul=explanation["prediction"],
        base_value=explanation["base_value"],
        model_used="XGBoost",
        feature_contributions=contributions,
        status="success",
    )
