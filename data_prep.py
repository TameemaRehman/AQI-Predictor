import pandas as pd
import numpy as np

DEFAULT_FEATURE_COLS = [
    # Raw pollutants
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    # Time features
    "hour", "day", "month", "day_of_week", "is_weekend",
    # Engineered features
    "aqi_change", "aqi_lag_1h", "aqi_lag_24h", "pm2_5_lag_24h",
    "aqi_roll_24h", "pm2_5_roll_24h"
]

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning appropriate for API time-series AQI data.
    - ensures timestamp dtype and ordering
    - removes duplicates
    - enforces basic sanity filters
    """
    df = df.copy()

    # Timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort + de-dupe
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="last")

    # Sanity filters
    if "aqi" in df.columns:
        df = df[df["aqi"].between(1, 5)]  # OpenWeather AQI category range

    # Pollutants should be non-negative (allow NaNs)
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    for c in pollutant_cols:
        if c in df.columns:
            df = df[(df[c].isna()) | (df[c] >= 0)]

    return df

def make_supervised_dataset(
    df: pd.DataFrame,
    horizon_hours: int = 72,
    feature_cols=None,
    target_col: str = "target_aqi_t_plus_72h",
):
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    df = df.copy()

    # Create target (future AQI)
    df[target_col] = df.groupby("location_id")["aqi"].shift(-horizon_hours)

    # Convert features to numeric (handles object/string types)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Drop rows missing any feature or target (includes NaNs from lags/rolling and coercion)
    train_df = df.dropna(subset=feature_cols + [target_col]).copy()

    X = train_df[feature_cols]
    y = train_df[target_col].astype(float)

    return train_df, X, y, feature_cols, target_col



def make_daily_actual_pred_df(df_with_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates hourly actual/pred to daily averages for non-technical charts.
    Expects columns: timestamp, actual, predicted
    """
    out = df_with_preds.copy()
    out["date"] = out["timestamp"].dt.date

    daily = (
        out.groupby("date")[["actual", "predicted"]]
        .mean()
        .reset_index()
    )
    daily["abs_error"] = (daily["actual"] - daily["predicted"]).abs()
    return daily