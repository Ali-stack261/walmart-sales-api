"""
Streamlit UI for the Walmart Sales Forecasting API.

Talks to the deployed FastAPI service (see api/main.py) over HTTP.
Configure the API base URL via the API_URL environment variable /
Streamlit secret, or edit the default below.
"""

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL") or "http://localhost:8000"

try:
    API_URL = os.environ.get("API_URL") or st.secrets["API_URL"]
except Exception:
    API_URL = os.environ.get("API_URL") or "http://localhost:8000"

st.set_page_config(page_title="Walmart Sales Forecast", page_icon="🛒", layout="wide")

st.title("🛒 Walmart Weekly Sales Forecast")
st.caption(f"Connected to API: `{API_URL}`")

# ── Sidebar: API health ─────────────────────────────────────────────
with st.sidebar:
    st.header("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            data = health.json()
            st.success("API is online")
            st.write(f"**Model:** {data.get('model', 'n/a')}")
            st.write(f"**Target:** {data.get('target', 'n/a')}")
        else:
            st.error(f"API returned status {health.status_code}")
    except requests.exceptions.RequestException as exc:
        st.error("Could not reach the API")
        st.caption(str(exc))
        st.info(
            "If this is a free-tier host (e.g. Render), the API may be "
            "asleep — the first request can take 30-60s to wake it up."
        )

    st.divider()
    st.header("About")
    st.write(
        "This app sends store/week details to a FastAPI model-serving "
        "endpoint and displays the predicted weekly sales."
    )

# ── Tabs: single prediction / batch / drift ─────────────────────────
tab_single, tab_batch, tab_drift = st.tabs(["Single Prediction", "Batch Prediction", "Drift Report"])

# ── Tab 1: single prediction ────────────────────────────────────────
with tab_single:
    st.subheader("Predict weekly sales for one store/week")

    col1, col2, col3 = st.columns(3)

    with col1:
        store = st.number_input("Store", min_value=1, max_value=45, value=1, step=1)
        holiday_flag = st.selectbox("Holiday week?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        pred_date = st.date_input("Week of", value=date(2024, 8, 1))

    with col2:
        temperature = st.slider("Temperature (°F)", min_value=-10.0, max_value=100.0, value=70.0)
        fuel_price = st.slider("Fuel Price ($)", min_value=2.0, max_value=5.0, value=3.4, step=0.01)

    with col3:
        cpi = st.slider("CPI", min_value=120.0, max_value=230.0, value=210.0)
        unemployment = st.slider("Unemployment (%)", min_value=3.0, max_value=15.0, value=7.5, step=0.1)

    if st.button("Predict", type="primary"):
        payload = {
            "Store": store,
            "Holiday_Flag": holiday_flag,
            "Temperature": temperature,
            "Fuel_Price": fuel_price,
            "CPI": cpi,
            "Unemployment": unemployment,
            "Month": pred_date.month,
            "WeekOfYear": pred_date.isocalendar()[1],
            "Year": pred_date.year,
        }
        try:
            with st.spinner("Predicting..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.metric("Predicted Weekly Sales", f"${result['prediction']:,.2f}")
                st.caption(f"Model: {result['model']} | Target: {result['target']}")
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")

# ── Tab 2: batch prediction ─────────────────────────────────────────
with tab_batch:
    st.subheader("Predict sales for multiple rows via CSV")
    st.caption(
        "Upload a CSV with columns: Store, Holiday_Flag, Temperature, "
        "Fuel_Price, CPI, Unemployment, Month, WeekOfYear, Year"
    )

    template_df = pd.DataFrame(
        [
            {
                "Store": 1, "Holiday_Flag": 0, "Temperature": 70.0, "Fuel_Price": 3.4,
                "CPI": 210.0, "Unemployment": 7.5, "Month": 8, "WeekOfYear": 32, "Year": 2024,
            }
        ]
    )
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df, use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            payload = df.to_dict(orient="records")
            try:
                with st.spinner(f"Predicting {len(payload)} rows..."):
                    response = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    df["Predicted_Sales"] = [p["prediction"] for p in predictions]
                    st.success(f"Predicted {len(df)} rows")
                    st.dataframe(df, use_container_width=True)

                    fig = go.Figure(data=[go.Bar(x=df.index.astype(str), y=df["Predicted_Sales"])])
                    fig.update_layout(
                        title="Predicted Sales by Row",
                        xaxis_title="Row",
                        yaxis_title="Predicted Weekly Sales ($)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "Download results CSV",
                        data=df.to_csv(index=False),
                        file_name="predictions_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

# ── Tab 3: drift report ─────────────────────────────────────────────
with tab_drift:
    st.subheader("Data drift summary")
    st.caption("Compares recent prediction inputs against the training data distribution.")

    if st.button("Fetch drift report"):
        try:
            with st.spinner("Fetching..."):
                response = requests.get(f"{API_URL}/drift", timeout=30)
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
