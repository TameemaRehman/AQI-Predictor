import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hopsworks

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

if not OPENWEATHER_API_KEY:
    raise ValueError("Missing OPENWEATHER_API_KEY in .env")
if not HOPSWORKS_PROJECT or not HOPSWORKS_API_KEY:
    raise ValueError("Missing HOPSWORKS_PROJECT or HOPSWORKS_API_KEY in .env")


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

            # Raw pollutants (store all available)
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),

            # AQI category (1-5) from OpenWeather
            "aqi": item.get("main", {}).get("aqi"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No records returned by OpenWeather API (empty dataframe).")

    # Ensure UTC-aware datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort for time-series features
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon..6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Simple change feature
    df["aqi_change"] = df.groupby("location_id")["aqi"].diff()

    # Lag features (strong baselines)
    df["aqi_lag_1h"] = df.groupby("location_id")["aqi"].shift(1)
    df["aqi_lag_24h"] = df.groupby("location_id")["aqi"].shift(24)
    df["pm2_5_lag_24h"] = df.groupby("location_id")["pm2_5"].shift(24)

    # Rolling features (24h)
    df["aqi_roll_24h"] = (
        df.groupby("location_id")["aqi"]
        .rolling(window=24, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["pm2_5_roll_24h"] = (
        df.groupby("location_id")["pm2_5"]
        .rolling(window=24, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Basic cleanup: keep numeric columns numeric
    numeric_cols = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi", "hour", "day", "month", "day_of_week", "is_weekend",
        "aqi_change", "aqi_lag_1h", "aqi_lag_24h", "pm2_5_lag_24h",
        "aqi_roll_24h", "pm2_5_roll_24h"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove duplicates just in case
    df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="last")

    return df


def upsert_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features_karachi",
        version=1,
        primary_key=["location_id", "timestamp"],
        event_time="timestamp",
        description="Karachi hourly AQI + pollutant features from OpenWeather (raw + engineered)."
    )

    fg.insert(df, write_options={"upsert": True})
    print(f"✅ Upsert complete: {len(df)} rows into Feature Group {fg.name}_v{fg.version}")


def main():
    print("Fetching OpenWeather history for Karachi...")
    data = fetch_openweather_history(LAT, LON, OPENWEATHER_API_KEY, days=120)
    print(f"✅ Fetched records: {len(data.get('list', []))}")

    print("Building feature dataframe...")
    df = json_to_features(data, LOCATION_ID)
    print("✅ Dataframe ready:", df.shape)
    print(df.head(3))

    # Sanity checks
    dupes = df.duplicated(subset=["location_id", "timestamp"]).sum()
    print("Duplicates:", dupes)
    print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())

    print("Writing to Hopsworks Feature Store...")
    upsert_to_hopsworks(df)

    print("Done.")


if __name__ == "__main__":
    main()
