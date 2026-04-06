"""
Streamlit Dashboard — CoreGuard Predictive RUL Engine.

This is the visual front-end. It provides a web page where users can:
    1. Enter sensor readings using sliders
    2. Click "Predict" to get the predicted Remaining Useful Life
    3. See a health status gauge (green/yellow/red)
    4. Click "Explain Prediction" to see which sensors drove the prediction
    5. View global SHAP feature importance plots

The dashboard does NOT load models directly.
It talks to the FastAPI backend (Phase 4) over HTTP.
This keeps the architecture clean: dashboard = frontend, API = backend.

To run this dashboard:
    1. Start the API first:   python scripts/serve.py
    2. Then start dashboard:  streamlit run dashboard/app.py
"""

import sys
import os

# adding project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import plotly.graph_objects as go
from pathlib import Path

from src.config import API_HOST, API_PORT, SHAP_OUTPUT_DIR

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CoreGuard — RUL Prediction Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API URL
# ---------------------------------------------------------------------------
# the FastAPI backend runs on this address
API_BASE_URL = f"http://localhost:{API_PORT}"


# ---------------------------------------------------------------------------
# SENSOR DEFINITIONS
# ---------------------------------------------------------------------------
# each sensor has: display name, min value, max value, default (median) value
# these ranges come from the actual NASA C-MAPSS FD001 training data
SENSOR_CONFIG = {
    "setting_1": {
        "label": "Setting 1 (Altitude)",
        "min": -0.0090, "max": 0.0090, "default": 0.0000, "step": 0.0001,
    },
    "setting_2": {
        "label": "Setting 2 (Mach Number)",
        "min": -0.0007, "max": 0.0007, "default": 0.0000, "step": 0.0001,
    },
    "sensor_2": {
        "label": "Sensor 2 — Total temp at LPC outlet",
        "min": 641.0, "max": 645.0, "default": 642.64, "step": 0.01,
    },
    "sensor_3": {
        "label": "Sensor 3 — Total temp at HPC outlet",
        "min": 1571.0, "max": 1617.0, "default": 1590.10, "step": 0.1,
    },
    "sensor_4": {
        "label": "Sensor 4 — Total temp at LPT outlet",
        "min": 1382.0, "max": 1442.0, "default": 1408.04, "step": 0.1,
    },
    "sensor_7": {
        "label": "Sensor 7 — Total pressure at HPC outlet",
        "min": 549.0, "max": 557.0, "default": 553.44, "step": 0.01,
    },
    "sensor_8": {
        "label": "Sensor 8 — Physical fan speed",
        "min": 2387.0, "max": 2389.0, "default": 2388.09, "step": 0.01,
    },
    "sensor_9": {
        "label": "Sensor 9 — Physical core speed",
        "min": 9021.0, "max": 9245.0, "default": 9060.66, "step": 0.1,
    },
    "sensor_11": {
        "label": "Sensor 11 — Static pressure at HPC outlet",
        "min": 46.80, "max": 48.60, "default": 47.51, "step": 0.01,
    },
    "sensor_12": {
        "label": "Sensor 12 — Fuel flow / Ps30 ratio",
        "min": 518.0, "max": 524.0, "default": 521.48, "step": 0.01,
    },
    "sensor_13": {
        "label": "Sensor 13 — Corrected fan speed",
        "min": 2387.0, "max": 2389.0, "default": 2388.09, "step": 0.01,
    },
    "sensor_14": {
        "label": "Sensor 14 — Corrected core speed",
        "min": 8099.0, "max": 8294.0, "default": 8140.54, "step": 0.1,
    },
    "sensor_15": {
        "label": "Sensor 15 — Bypass ratio",
        "min": 8.32, "max": 8.59, "default": 8.44, "step": 0.001,
    },
    "sensor_17": {
        "label": "Sensor 17 — Bleed enthalpy",
        "min": 388.0, "max": 400.0, "default": 393.0, "step": 0.1,
    },
    "sensor_20": {
        "label": "Sensor 20 — HPT coolant bleed",
        "min": 38.10, "max": 39.50, "default": 38.83, "step": 0.01,
    },
    "sensor_21": {
        "label": "Sensor 21 — LPT coolant bleed",
        "min": 22.89, "max": 23.62, "default": 23.30, "step": 0.001,
    },
}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def check_api_health() -> bool:
    """Ping the API health endpoint. Returns True if the API is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except requests.ConnectionError:
        return False


def get_prediction(sensor_data: dict) -> dict:
    """Call the /predict endpoint. Returns the response dict."""
    resp = requests.post(f"{API_BASE_URL}/predict", json=sensor_data, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_explanation(sensor_data: dict) -> dict:
    """Call the /predict/explain endpoint. Returns the response dict."""
    resp = requests.post(f"{API_BASE_URL}/predict/explain", json=sensor_data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def create_rul_gauge(predicted_rul: float) -> go.Figure:
    """
    Create a gauge chart showing the predicted RUL with color zones.

    Color zones:
        0-30 cycles   → RED    (critical — failure is very close)
        30-70 cycles  → YELLOW (warning — schedule maintenance soon)
        70-125 cycles → GREEN  (healthy — no immediate action needed)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_rul,
        title={"text": "Predicted Remaining Useful Life (Cycles)", "font": {"size": 20}},
        number={"font": {"size": 48}, "suffix": " cycles"},
        gauge={
            "axis": {"range": [0, 130], "tickwidth": 2},
            "bar": {"color": "#1f77b4", "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#ff4d4d"},
                {"range": [30, 70], "color": "#ffd633"},
                {"range": [70, 130], "color": "#4dff4d"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": predicted_rul,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def create_shap_bar_chart(contributions: list, top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart from SHAP feature contributions.

    Red bars = pushed prediction DOWN (closer to failure).
    Green bars = pushed prediction UP (more remaining life).
    """
    # take only top N contributors
    top = contributions[:top_n]

    # reverse so the most important is at the top of the chart
    top = list(reversed(top))

    features = [c["feature"] for c in top]
    values = [c["shap_value"] for c in top]
    colors = ["#4dff4d" if v > 0 else "#ff4d4d" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Sensor Contributions to This Prediction (SHAP Values)",
        xaxis_title="Impact on Predicted RUL (cycles)",
        yaxis_title="",
        height=400,
        margin=dict(l=180, r=40, t=50, b=40),
    )
    return fig


def get_health_status(rul: float) -> tuple:
    """
    Return a status label and color based on predicted RUL.

    Returns:
        Tuple of (status_text, color_hex)
    """
    if rul <= 30:
        return "CRITICAL", "#ff4d4d"
    elif rul <= 70:
        return "WARNING", "#ffd633"
    else:
        return "HEALTHY", "#4dff4d"


# ---------------------------------------------------------------------------
# MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------------------------
def main():
    """Build the full Streamlit dashboard."""

    # --- HEADER ---
    st.title("CoreGuard — Predictive RUL Engine")
    st.markdown(
        "Predict Remaining Useful Life of turbofan engines using sensor data. "
        "Powered by XGBoost with SHAP explainability."
    )

    # --- API STATUS CHECK ---
    api_healthy = check_api_health()

    if not api_healthy:
        st.error(
            "**API is not running.** Start the backend first with: "
            "`python scripts/serve.py`"
        )
        st.stop()

    st.success("API connected and model loaded.")

    # --- SIDEBAR: SENSOR INPUTS ---
    st.sidebar.header("Sensor Readings")
    st.sidebar.markdown("Adjust sensor values below. Click **Predict** to get the RUL.")

    sensor_values = {}
    for key, config in SENSOR_CONFIG.items():
        sensor_values[key] = st.sidebar.slider(
            label=config["label"],
            min_value=float(config["min"]),
            max_value=float(config["max"]),
            value=float(config["default"]),
            step=float(config["step"]),
            key=key,
        )

    # --- PREDICTION SECTION ---
    st.markdown("---")

    col_predict, col_explain = st.columns(2)

    with col_predict:
        predict_clicked = st.button("Predict RUL", type="primary", use_container_width=True)

    with col_explain:
        explain_clicked = st.button("Explain Prediction", use_container_width=True)

    # --- RESULTS ---
    if predict_clicked or explain_clicked:

        # always get the prediction first
        try:
            if explain_clicked:
                result = get_explanation(sensor_values)
            else:
                result = get_prediction(sensor_values)
        except requests.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

        predicted_rul = result["predicted_rul"]
        status_text, status_color = get_health_status(predicted_rul)

        # --- RUL GAUGE ---
        st.markdown("---")
        st.subheader("Prediction Result")

        col_gauge, col_status = st.columns([2, 1])

        with col_gauge:
            gauge_fig = create_rul_gauge(predicted_rul)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_status:
            st.markdown(f"### Engine Status")
            st.markdown(
                f'<div style="background-color:{status_color}; padding:20px; '
                f'border-radius:10px; text-align:center;">'
                f'<h1 style="color:black; margin:0;">{status_text}</h1>'
                f'<p style="color:black; margin:5px 0 0 0; font-size:18px;">'
                f'{predicted_rul:.1f} cycles remaining</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.metric("Model Used", result.get("model_used", "XGBoost"))

        # --- SHAP EXPLANATION (only if explain was clicked) ---
        if explain_clicked and "feature_contributions" in result:
            st.markdown("---")
            st.subheader("Prediction Explanation (SHAP)")

            st.markdown(
                f"**Base value:** {result['base_value']:.2f} cycles "
                f"(average prediction across all training engines)"
            )

            # bar chart of top contributions
            bar_fig = create_shap_bar_chart(result["feature_contributions"], top_n=10)
            st.plotly_chart(bar_fig, use_container_width=True)

            # expandable table of all contributions
            with st.expander("View all 44 feature contributions"):
                for i, c in enumerate(result["feature_contributions"], 1):
                    direction = "+" if c["shap_value"] > 0 else ""
                    st.text(
                        f"{i:>2}. {c['feature']:<25} "
                        f"value={c['value']:.4f}  "
                        f"impact={direction}{c['shap_value']:.4f} cycles"
                    )

    # --- GLOBAL SHAP SECTION ---
    st.markdown("---")
    st.subheader("Global Feature Importance")
    st.markdown(
        "These plots show which sensors matter most **across all engines**, "
        "not just one specific prediction."
    )

    tab_summary, tab_bar = st.tabs(["Detailed Summary", "Simple Bar Chart"])

    summary_path = SHAP_OUTPUT_DIR / "shap_summary.png"
    bar_path = SHAP_OUTPUT_DIR / "shap_bar_importance.png"

    with tab_summary:
        if summary_path.exists():
            st.image(str(summary_path), caption="SHAP Summary Plot — each dot is one engine")
        else:
            st.warning(
                "Summary plot not found. Run `python -m src.explainability.shap_explainer` "
                "to generate it."
            )

    with tab_bar:
        if bar_path.exists():
            st.image(str(bar_path), caption="Average Sensor Impact on RUL Prediction")
        else:
            st.warning(
                "Bar plot not found. Run `python -m src.explainability.shap_explainer` "
                "to generate it."
            )

    # --- FOOTER ---
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:12px;'>"
        "CoreGuard Predictive RUL Engine — Built with XGBoost, SHAP, FastAPI, Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
