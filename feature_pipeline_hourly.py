import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))
LOCATION_ID = os.getenv("LOCATION_ID", "karachi")

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FG_NAME = "aqi_features_karachi"
FG_VERSION = 1


def get_feature_group(project):
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["location_id", "timestamp"],
        event_time="timestamp",
        description="Karachi hourly AQI + pollutant features from OpenWeather (raw + engineered).",
    )
    return fg


def get_last_timestamp(fg) -> datetime:
    """
    Read the latest timestamp from the feature group.
    If none exists, fallback to last 7 days (or 120 days for initial backfill).
    """
    df = fg.read()  # for 3k rows this is fine; later you can optimize via SQL query
    if df.empty:
        return datetime.now(timezone.utc) - timedelta(days=7)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df["timestamp"].max()


def fetch_openweather_history(lat, lon, api_key, start_ts: int, end_ts: int) -> dict:
    url = (
        "https://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}"
    )
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenWeather error {resp.status_code}: {resp.text}")
    return resp.json()


def json_to_features(data: dict, location_id: str) -> pd.DataFrame:
    rows = []
    for item in data.get("list", []):
        comp = item.get("components", {})
        rows.append({
            "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
            "location_id": location_id,
            "co": comp.get("co"),
            "no": comp.get("no"),
            "no2": comp.get("no2"),
            "o3": comp.get("o3"),
            "so2": comp.get("so2"),
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "nh3": comp.get("nh3"),
            "aqi": item.get("main", {}).get("aqi"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Engineered features (needs history; for small increments, lags may be NaN unless you include overlap)
    df["aqi_change"] = df.groupby("location_id")["aqi"].diff()
    df["aqi_lag_1h"] = df.groupby("location_id")["aqi"].shift(1)
    df["aqi_lag_24h"] = df.groupby("location_id")["aqi"].shift(24)
    df["pm2_5_lag_24h"] = df.groupby("location_id")["pm2_5"].shift(24)

    df["aqi_roll_24h"] = (
        df.groupby("location_id")["aqi"]
        .rolling(window=24, min_periods=12).mean()
        .reset_index(level=0, drop=True)
    )
    df["pm2_5_roll_24h"] = (
        df.groupby("location_id")["pm2_5"]
        .rolling(window=24, min_periods=12).mean()
        .reset_index(level=0, drop=True)
    )

    # Ensure numeric
    numeric_cols = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi", "hour", "day", "month", "day_of_week", "is_weekend",
        "aqi_change", "aqi_lag_1h", "aqi_lag_24h", "pm2_5_lag_24h",
        "aqi_roll_24h", "pm2_5_roll_24h",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fg = get_feature_group(project)

    last_ts = get_last_timestamp(fg)
    # Include overlap of 24h so lags/rolling compute correctly for new rows
    start_dt = last_ts - timedelta(hours=24)
    end_dt = datetime.now(timezone.utc)

    start_ts = int(start_dt.timestamp()) + 1
    end_ts = int(end_dt.timestamp())

    print(f"Last timestamp in FG: {last_ts}")
    print(f"Fetching from {start_dt} to {end_dt} ...")

    data = fetch_openweather_history(LAT, LON, OPENWEATHER_API_KEY, start_ts, end_ts)
    df = json_to_features(data, LOCATION_ID)

    if df.empty:
        print("No new data returned from API.")
        return

    # Keep only truly new rows for insertion (avoid unnecessary upserts)
    df = df[df["timestamp"] > last_ts].copy()
    if df.empty:
        print("No rows newer than last timestamp. Nothing to insert.")
        return

    fg.insert(df, write_options={"upsert": True})
    print(f"✅ Inserted/Upserted {len(df)} new rows.")


if __name__ == "__main__":
    main()
