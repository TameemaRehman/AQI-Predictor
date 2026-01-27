

# import requests
# from datetime import datetime, timedelta, timezone
# from dotenv import load_dotenv
# import os

# load_dotenv()

# API_KEY = os.getenv("OPENWEATHER_API_KEY")
# LAT = os.getenv("LAT")
# LON = os.getenv("LON")

# end_time = int(datetime.now(timezone.utc).timestamp())
# start_time = int((datetime.now(timezone.utc) - timedelta(days=120)).timestamp())

# url = (
#     f"https://api.openweathermap.org/data/2.5/air_pollution/history"
#     f"?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"
# )

# response = requests.get(url)
# print(response.status_code)
# print(response.json())  # if 200, JSON will contain your data

import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import pandas as pd

# Load .env
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))

# Calculate timestamps (past 4 months)
end_time = int(datetime.now(timezone.utc).timestamp())
start_time = int((datetime.now(timezone.utc) - timedelta(days=120)).timestamp())

# Build URL
url = (
    f"https://api.openweathermap.org/data/2.5/air_pollution/history"
    f"?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={API_KEY}"
)

# Fetch data
response = requests.get(url)

if response.status_code != 200:
    print(f"Error: {response.status_code}, {response.json()}")
else:
    data = response.json()
    print(f"Data fetched successfully: {len(data['list'])} records")

    # Convert JSON to DataFrame
    rows = []
    for item in data["list"]:
        rows.append({
            "timestamp": datetime.utcfromtimestamp(item["dt"]),
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "no2": item["components"]["no2"],
            "so2": item["components"]["so2"],
            "co": item["components"]["co"],
            "o3": item["components"]["o3"],
            "aqi": item["main"]["aqi"]
        })

    df = pd.DataFrame(rows)

    # Add time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["location_id"] = "karachi"

    # Compute AQI change rate
    df["aqi_change"] = df["aqi"].diff().fillna(0)

    # Check the dataframe
    print(df.head())
    print(df.info())
    print(df.describe())
