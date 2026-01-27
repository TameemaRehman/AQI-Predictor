import os
import time
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
    df = fg.read()
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
    df["location_id"] = df["location_id"].astype(str)

    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5)  # bool for now

    # Engineered features
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

    # Float-like columns
    float_cols = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi_change", "aqi_lag_1h", "aqi_lag_24h", "pm2_5_lag_24h",
        "aqi_roll_24h", "pm2_5_roll_24h",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Match Feature Group schema types
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype("int64")
    for c in ["hour", "day", "month", "day_of_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")
    df["is_weekend"] = df["is_weekend"].astype("int32")

    # Remove duplicates on primary key
    df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="last")
    return df


def insert_with_retries(fg, df: pd.DataFrame, max_retries: int = 5, base_sleep: int = 10):
    """
    Hopsworks/GitHub runners sometimes drop the connection when the materialization job starts.
    This retries transient network errors with exponential-ish backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df, write_options={"upsert": True})
            return
        except Exception as e:
            msg = str(e).lower()
            retryable = (
                "remote end closed connection" in msg
                or "connection aborted" in msg
                or "connectionerror" in msg
                or "timed out" in msg
                or "temporarily unavailable" in msg
                or "protocolerror" in msg
            )
            if (not retryable) or (attempt == max_retries):
                raise

            sleep_s = base_sleep * attempt
            print(f"⚠️ Insert failed (attempt {attempt}/{max_retries}). Retrying in {sleep_s}s...")
            time.sleep(sleep_s)


def chunked_insert(fg, df: pd.DataFrame, chunk_size: int = 200):
    """
    More robust than a single insert if Hopsworks connection is flaky.
    Uses upsert so reruns won't duplicate.
    """
    n = len(df)
    for i in range(0, n, chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        print(f"Inserting chunk {i+1}-{i+len(chunk)} of {n}...")
        insert_with_retries(fg, chunk)


def main():
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fg = get_feature_group(project)

    last_ts = get_last_timestamp(fg)

    # overlap for lag/rolling
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

    # insert only new rows
    df = df[df["timestamp"] > last_ts].copy()
    if df.empty:
        print("No rows newer than last timestamp. Nothing to insert.")
        return

    # Final safety casting (schema expectations)
    df["location_id"] = df["location_id"].astype(str)
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype("int64")
    for c in ["hour", "day", "month", "day_of_week", "is_weekend"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")

    # ✅ Use chunked + retry insert to prevent transient connection failures
    chunked_insert(fg, df, chunk_size=200)

    print(f"✅ Inserted/Upserted {len(df)} new rows.")


if __name__ == "__main__":
    main()
