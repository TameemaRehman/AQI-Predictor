import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import hopsworks
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_prep import basic_cleaning, make_supervised_dataset, make_daily_actual_pred_df

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FG_NAME = "aqi_features_karachi"
FG_VERSION = 1
HORIZON_HOURS = 24

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_time_series(model, X, y, n_splits=3):
    tss = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []
    last_split_payload = None

    for train_idx, test_idx in tss.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmses.append(rmse(y_test, preds))
        maes.append(float(mean_absolute_error(y_test, preds)))
        r2s.append(float(r2_score(y_test, preds)))

        last_split_payload = {
            "test_idx": test_idx,
            "y_test": y_test.reset_index(drop=True),
            "preds": pd.Series(preds).reset_index(drop=True),
        }

    return {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
        "last_split": last_split_payload,
    }


def register_model(project, model_name, model_path, metrics, description):
    mr = project.get_model_registry()
    m = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=description,
    )
    m.save(model_path)
    print(f"✅ Registered: {model_name} v{m.version}")
    return m


def load_feature_group_df():
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()
    return project, df


def main():
    print("Loading from Hopsworks Feature Group...")
    project, df = load_feature_group_df()
    print("Raw shape:", df.shape)

    print("Cleaning (training-time)...")
    df = basic_cleaning(df)
    print("Cleaned shape:", df.shape)

    print("Building supervised dataset (72h ahead target)...")
    train_df, X, y, feature_cols, target_col = make_supervised_dataset(df, horizon_hours=HORIZON_HOURS)
    print("Training rows:", X.shape[0], "Features:", X.shape[1])
    print("Time range:", train_df["timestamp"].min(), "->", train_df["timestamp"].max())

    models = [
        ("ridge",
         Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
         "Ridge Regression"),
        ("rf",
         RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
         "Random Forest"),
        ("gbr",
         GradientBoostingRegressor(n_estimators=400, random_state=42),
         "Gradient Boosting"),
    ]

    results = []
    best = None

    # Train + evaluate + register EACH model
    for key, model, pretty_name in models:
        print(f"\nTraining/Evaluating: {pretty_name}")
        metrics = evaluate_time_series(model, X, y, n_splits=3)

        row = {
            "model_key": key,
            "model_name": pretty_name,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
        results.append(row)

        # Fit on full data (so registry artifact is fully-trained)
        model.fit(X, y)

        # Save local artifact (consistent file name per model)
        model_path = os.path.join(ARTIFACT_DIR, f"{key}_model.pkl")
        joblib.dump(model, model_path)

        # Register in Hopsworks Model Registry
        register_model(
            project=project,
            model_name=f"aqi_{key}_72h",
            model_path=model_path,
            metrics={"rmse": row["rmse"], "mae": row["mae"], "r2": row["r2"]},
            description=f"{pretty_name} model for Karachi AQI prediction 72h ahead.",
        )

        # Track best model (by RMSE)
        if best is None or row["rmse"] < best["rmse"]:
            best = {
                **row,
                "model_obj": model,
                "model_path": model_path,
                "last_split": metrics["last_split"],
            }

    # Results table
    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print("\n=== Model Comparison (lower RMSE better) ===")
    print(results_df)

    # Save shared artifacts for Streamlit/report
    with open(os.path.join(ARTIFACT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(
            {
                "best_model_key": best["model_key"],
                "best_model_name": best["model_name"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "r2": float(best["r2"]),
                "horizon_hours": HORIZON_HOURS,
                "n_rows_train": int(X.shape[0]),
            },
            f,
            indent=2,
        )

    # Create daily validation plot data (non-technical)
    last = best["last_split"]
    test_idx = last["test_idx"]
    test_rows = train_df.iloc[test_idx].copy()
    test_rows["actual"] = last["y_test"].values
    test_rows["predicted"] = last["preds"].values

    hourly_path = os.path.join(ARTIFACT_DIR, "last_split_predictions_hourly.csv")
    test_rows[["timestamp", "actual", "predicted"]].to_csv(hourly_path, index=False)

    daily = make_daily_actual_pred_df(test_rows[["timestamp", "actual", "predicted"]])
    daily_path = os.path.join(ARTIFACT_DIR, "last_split_predictions_daily.csv")
    daily.to_csv(daily_path, index=False)

    # Register champion model (points to best model artifact)
    print("\nRegistering CHAMPION model...")
    register_model(
        project=project,
        model_name="aqi_champion_72h",
        model_path=best["model_path"],
        metrics={"rmse": float(best["rmse"]), "mae": float(best["mae"]), "r2": float(best["r2"])},
        description=f"Champion model selected among Ridge/RF/GBR. Winner: {best['model_name']}.",
    )

    print("\n✅ Done. Artifacts saved to:", ARTIFACT_DIR)
    print("Champion:", best["model_name"])
    print("Daily plot data:", daily_path)


if __name__ == "__main__":
    main()
