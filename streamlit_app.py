import os
import json
import pandas as pd
import joblib
import streamlit as st
from dotenv import load_dotenv
import hopsworks

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FG_NAME = "aqi_features_karachi"
FG_VERSION = 1

ARTIFACT_DIR = "artifacts"
CHAMPION_MODEL_NAME = "aqi_champion_24h"

st.set_page_config(page_title="Karachi AQI Forecast (3 Days)", layout="wide")


@st.cache_resource
def hopsworks_login():
    return hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)


@st.cache_resource
def load_champion_model_from_registry(model_name: str = CHAMPION_MODEL_NAME):
    """
    Downloads latest version of model from Hopsworks Model Registry and loads it with joblib.
    """
    project = hopsworks_login()
    mr = project.get_model_registry()

    # latest version
    m = mr.get_model(model_name, version=None)
    model_dir = m.download()

    # Our training pipeline saved champion artifact as either ridge_model.pkl/rf_model.pkl/gbr_model.pkl.
    # Champion points to one of those paths. We'll load whichever .pkl exists in the downloaded folder.
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("No .pkl found in downloaded model directory from Hopsworks registry.")
    # If multiple, pick the first
    model_path = os.path.join(model_dir, pkl_files[0])
    model = joblib.load(model_path)
    return model


def load_feature_cols_and_metrics():
    feature_cols = None
    metrics = None

    fc_path = os.path.join(ARTIFACT_DIR, "feature_cols.json")
    mt_path = os.path.join(ARTIFACT_DIR, "metrics.json")

    if os.path.exists(fc_path):
        with open(fc_path) as f:
            feature_cols = json.load(f)

    if os.path.exists(mt_path):
        with open(mt_path) as f:
            metrics = json.load(f)

    return feature_cols, metrics


@st.cache_data(ttl=300)
def load_feature_store_data():
    project = hopsworks_login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)

    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    latest = df.tail(1).copy()
    return df, latest


def aqi_label(aqi_value):
    mapping = {
        1: ("Good", "🟢"),
        2: ("Fair", "🟡"),
        3: ("Moderate", "🟠"),
        4: ("Poor", "🔴"),
        5: ("Very Poor", "🟣"),
    }
    v = int(round(float(aqi_value)))
    v = max(1, min(5, v))
    return mapping[v]


# ---------------- UI ----------------
st.title("Karachi AQI Forecast (Next 3 Days)")

feature_cols, metrics = load_feature_cols_and_metrics()
model = load_champion_model_from_registry()

if metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion Model", metrics.get("best_model_name", "aqi_champion_24h"))
    c2.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
    c3.metric("MAE", f"{metrics.get('mae', 0):.3f}")
    c4.metric("R²", f"{metrics.get('r2', 0):.3f}")
    st.caption(
        "Model scores are computed using time-series validation (no shuffling). "
        "Lower RMSE/MAE is better."
    )
else:
    st.info("metrics.json not found. Run training_pipeline.py to generate metrics artifacts.")

st.divider()

with st.spinner("Loading latest data from Feature Store..."):
    full_df, latest_row = load_feature_store_data()

latest_ts = latest_row["timestamp"].iloc[0]
current_aqi = latest_row["aqi"].iloc[0]
label, emoji = aqi_label(current_aqi)

st.subheader("Latest Observed AQI")
st.write(f"**Latest timestamp (UTC):** {latest_ts}")
st.write(f"**Current AQI (OpenWeather category):** {emoji} **{int(current_aqi)} — {label}**")

st.divider()

st.subheader("Prediction for ~72 hours ahead (3 days)")

if not feature_cols:
    st.error("feature_cols.json not found. Run training_pipeline.py to generate it.")
else:
    X_latest = latest_row[feature_cols].copy()

    # ensure numeric
    X_latest = X_latest.apply(pd.to_numeric, errors="coerce")
    if X_latest.isna().any(axis=None):
        st.warning("Latest row has missing values in required features. Try running aqi_predict.py again to refresh features.")

    pred_72h = float(model.predict(X_latest)[0])
    pred_label, pred_emoji = aqi_label(pred_72h)

    st.markdown(f"### Forecast AQI in ~3 days: {pred_emoji} **{int(round(pred_72h))} — {pred_label}**")
    st.caption("This is the predicted OpenWeather AQI category (1–5).")

st.divider()

st.subheader("Model Performance (Daily View — non-technical)")
daily_path = os.path.join(ARTIFACT_DIR, "last_split_predictions_daily.csv")

if os.path.exists(daily_path):
    daily = pd.read_csv(daily_path)
    daily["date"] = pd.to_datetime(daily["date"])
    st.line_chart(daily.set_index("date")[["actual", "predicted"]])
    st.caption("Daily average AQI: Actual vs Predicted (validation period).")

    st.bar_chart(daily.set_index("date")[["abs_error"]])
    st.caption("Daily absolute error (lower is better).")
else:
    st.warning("Daily prediction file not found. Run training_pipeline.py first to generate it.")

st.divider()

st.subheader("Last 30 Days AQI Trend (Observed)")
last30 = full_df.sort_values("timestamp").tail(24 * 30).copy()
last30["date"] = last30["timestamp"].dt.date
daily_obs = last30.groupby("date")["aqi"].mean().reset_index()
daily_obs["date"] = pd.to_datetime(daily_obs["date"])

st.line_chart(daily_obs.set_index("date")[["aqi"]])
st.caption("Daily average observed AQI for the last 30 days.")
