"""
Pydantic Schemas for the CoreGuard API.

Pydantic is a data validation library. It defines the exact shape
of data going INTO the API (requests) and coming OUT of the API (responses).

Why we need schemas:
    Without schemas, the API would accept anything — bad data, missing fields,
    wrong types. Pydantic checks every incoming request automatically.
    If something is wrong, it returns a clear error message telling the client
    exactly what field is missing or what type is expected.

Three schema groups here:
    1. SensorInput — what the client sends (sensor readings for one engine)
    2. PredictionResponse — what the API returns (predicted RUL)
    3. ExplanationResponse — what the API returns when SHAP explanation is requested
"""

from typing import Optional
from pydantic import BaseModel, Field


class SensorInput(BaseModel):
    """
    Input schema: one engine's sensor readings at a single point in time.

    The client sends this JSON to the /predict or /predict/explain endpoint.
    Every field has a default value so the API still works even if a client
    omits some sensors. In production, all sensors would be provided.

    The field names match the column names in the processed training data.
    """

    # operational settings
    setting_1: float = Field(default=0.0, description="Operational setting 1 (altitude-related)")
    setting_2: float = Field(default=0.0, description="Operational setting 2 (Mach number-related)")

    # 14 useful sensors (same ones selected during preprocessing in Phase 1)
    sensor_2: float = Field(default=0.0, description="Total temperature at LPC outlet")
    sensor_3: float = Field(default=0.0, description="Total temperature at HPC outlet")
    sensor_4: float = Field(default=0.0, description="Total temperature at LPT outlet")
    sensor_7: float = Field(default=0.0, description="Total pressure at HPC outlet")
    sensor_8: float = Field(default=0.0, description="Physical fan speed")
    sensor_9: float = Field(default=0.0, description="Physical core speed")
    sensor_11: float = Field(default=0.0, description="Static pressure at HPC outlet")
    sensor_12: float = Field(default=0.0, description="Ratio of fuel flow to Ps30")
    sensor_13: float = Field(default=0.0, description="Corrected fan speed")
    sensor_14: float = Field(default=0.0, description="Corrected core speed")
    sensor_15: float = Field(default=0.0, description="Bypass ratio")
    sensor_17: float = Field(default=0.0, description="Bleed enthalpy")
    sensor_20: float = Field(default=0.0, description="HPT coolant bleed")
    sensor_21: float = Field(default=0.0, description="LPT coolant bleed")

    class Config:
        # this example appears in the auto-generated API docs (Swagger UI)
        json_schema_extra = {
            "example": {
                "setting_1": 0.0023,
                "setting_2": 0.0003,
                "sensor_2": 641.82,
                "sensor_3": 1589.7,
                "sensor_4": 1400.6,
                "sensor_7": 554.36,
                "sensor_8": 2388.0,
                "sensor_9": 9046.19,
                "sensor_11": 47.47,
                "sensor_12": 521.66,
                "sensor_13": 2388.02,
                "sensor_14": 8138.62,
                "sensor_15": 8.4195,
                "sensor_17": 392.0,
                "sensor_20": 39.06,
                "sensor_21": 23.419,
            }
        }


class PredictionResponse(BaseModel):
    """
    Output schema for the /predict endpoint.

    Returns just the predicted RUL — no SHAP explanation.
    This is the fast endpoint for when you just need the number.
    """

    predicted_rul: float = Field(description="Predicted Remaining Useful Life in cycles")
    model_used: str = Field(description="Which model made this prediction")
    status: str = Field(default="success", description="Request status")


class FeatureContribution(BaseModel):
    """
    One feature's contribution to the prediction.

    Used inside ExplanationResponse to show each sensor's SHAP value.
    """

    feature: str = Field(description="Feature name (e.g., sensor_4, rolling_sensor_9)")
    value: float = Field(description="The actual value of this feature in the input")
    shap_value: float = Field(description="How much this feature pushed the prediction (positive = more life, negative = less life)")


class ExplanationResponse(BaseModel):
    """
    Output schema for the /predict/explain endpoint.

    Returns the predicted RUL PLUS a full SHAP breakdown showing
    which sensors drove the prediction and by how much.
    """

    predicted_rul: float = Field(description="Predicted Remaining Useful Life in cycles")
    base_value: float = Field(description="Average prediction across all training data (baseline)")
    model_used: str = Field(description="Which model made this prediction")
    feature_contributions: list[FeatureContribution] = Field(
        description="Each sensor's contribution to the prediction, sorted by absolute impact"
    )
    status: str = Field(default="success", description="Request status")


class HealthResponse(BaseModel):
    """
    Output schema for the /health endpoint.

    Simple check to confirm the API is running and the model is loaded.
    """

    status: str = Field(description="API status")
    model_loaded: bool = Field(description="Whether the XGBoost model is loaded in memory")
    model_type: str = Field(description="Type of model loaded")
