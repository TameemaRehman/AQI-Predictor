import os
import json
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hopsworks

from data_prep import basic_cleaning, make_supervised_dataset, make_daily_actual_pred_df

# -------------------------
# Config
# -------------------------
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))
LOCATION_ID = os.getenv("LOCATION_ID", "karachi")

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

ARTIFACT_DIR = "artifacts"
CHAMPION_MODEL_FILE = os.path.join(ARTIFACT_DIR, "aqi_champion_24h_model.pkl")
FEATURE_COLS_FILE = os.path.join(ARTIFACT_DIR, "feature_cols.json")
HORIZON_HOURS = 1  # we will predict hour by hour for 72h

os.makedirs(ARTIFACT_DIR, exist_ok=True)

if not OPENWEATHER_API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY in .env")
if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
    raise ValueError("Missing HOPSWORKS_PROJECT or HOPSWORKS_API_KEY in .env")

# -------------------------
# Helper Functions
# -------------------------
def fetch_openweather_history(lat: float, lon: float, api_key: str, days: int = 120) -> dict:
    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_time}&end={end_time}&appid={api_key}"
    )
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenWeather error {resp.status_code}: {resp.text}")
    return resp.json()


def json_to_features(data: dict, location_id: str) -> pd.DataFrame:
    rows = []
    for item in data.get("list", []):
        components = item.get("components", {})
        rows.append({
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "location_id": location_id,
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
            "aqi": item.get("main", {}).get("aqi"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No records returned by OpenWeather API (empty dataframe).")

    # Time features
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Change, lag, rolling features
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    df["aqi_change"] = df.groupby("location_id")["aqi"].diff()
    df["aqi_lag_1h"] = df.groupby("location_id")["aqi"].shift(1)
    df["aqi_lag_24h"] = df.groupby("location_id")["aqi"].shift(24)
    df["pm2_5_lag_24h"] = df.groupby("location_id")["pm2_5"].shift(24)
    df["aqi_roll_24h"] = df.groupby("location_id")["aqi"].rolling(window=24, min_periods=12).mean().reset_index(level=0, drop=True)
    df["pm2_5_roll_24h"] = df.groupby("location_id")["pm2_5"].rolling(window=24, min_periods=12).mean().reset_index(level=0, drop=True)

    # Numeric cleanup
    for c in ["co","no","no2","o3","so2","pm2_5","pm10","nh3","aqi",
              "hour","day","month","day_of_week","is_weekend",
              "aqi_change","aqi_lag_1h","aqi_lag_24h","pm2_5_lag_24h",
              "aqi_roll_24h","pm2_5_roll_24h"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates(subset=["location_id","timestamp"], keep="last")
    return df


def upsert_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features_karachi",
        version=1,
        primary_key=["location_id","timestamp"],
        event_time="timestamp",
        description="Karachi hourly AQI + pollutant features"
    )
    fg.insert(df, write_options={"upsert": True})
    print(f"✅ Upsert complete: {len(df)} rows.")


def recursive_72h_forecast(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    latest_row = df.iloc[-1:].copy()
    forecasts = []

    for hour in range(1, 73):  # next 72 hours
        X = latest_row[feature_cols]
        pred = model.predict(X)[0]
        forecast_time = latest_row["timestamp"].values[0] + np.timedelta64(1, 'h')

        forecasts.append({
            "timestamp": forecast_time,
            "predicted_aqi": float(pred)
        })

        # Update latest_row for next iteration
        latest_row["aqi"] = pred
        latest_row["aqi_lag_1h"] = pred
        latest_row["aqi_roll_24h"] = pred

        latest_row["timestamp"] = forecast_time

    return pd.DataFrame(forecasts)


# -------------------------
# Main
# -------------------------
def main():
    print("Fetching OpenWeather history...")
    data = fetch_openweather_history(LAT, LON, OPENWEATHER_API_KEY, days=120)
    df = json_to_features(data, LOCATION_ID)
    df = basic_cleaning(df)
    upsert_to_hopsworks(df)

    # Load champion model + feature columns
    print("Loading champion model...")
    model = joblib.load(CHAMPION_MODEL_FILE)
    with open(FEATURE_COLS_FILE) as f:
        feature_cols = json.load(f)

    print("Predicting AQI for next 72 hours...")
    forecast_hourly = recursive_72h_forecast(df, model, feature_cols)
    forecast_hourly.to_csv(os.path.join(ARTIFACT_DIR, "aqi_3day_forecast_hourly.csv"), index=False)

    print("Aggregating to daily forecast...")
    forecast_hourly["timestamp"] = pd.to_datetime(forecast_hourly["timestamp"])
    forecast_daily = forecast_hourly.copy()
    forecast_daily["date"] = forecast_daily["timestamp"].dt.date
    forecast_daily = forecast_daily.groupby("date")["predicted_aqi"].mean().reset_index()
    forecast_daily.to_csv(os.path.join(ARTIFACT_DIR, "aqi_3day_forecast_daily.csv"), index=False)

    print("✅ Forecasting complete. Hourly & daily forecasts saved.")


if __name__ == "__main__":
    main()
