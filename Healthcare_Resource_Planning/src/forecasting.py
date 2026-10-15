"""Simple, explainable department-level occupied-bed forecasting."""
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["time_index", "weekday_sin", "weekday_cos", "lag_1", "lag_7", "rolling_7"]

def _features(series: pd.Series) -> pd.DataFrame:
    """Create time, weekday, previous-day, previous-week and rolling features."""
    y = series.astype(float).reset_index(drop=True)
    n = len(y)
    frame = pd.DataFrame({"time_index": np.arange(n, dtype=float)})
    weekdays = np.arange(n) % 7
    frame["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
    frame["lag_1"] = y.shift(1)
    frame["lag_7"] = y.shift(7)
    frame["rolling_7"] = y.shift(1).rolling(7, min_periods=3).mean()
    return frame

def _fit_model(values: pd.Series):
    y = values.astype(float).reset_index(drop=True)
    X = _features(y)
    valid = X.notna().all(axis=1)
    if valid.sum() < 14:
        raise ValueError("At least 14 usable daily observations are needed.")
    model = Ridge(alpha=2.0)
    model.fit(X.loc[valid, FEATURES], y.loc[valid])
    return model

def evaluate_department(group: pd.DataFrame, test_days: int = 30) -> dict:
    """Chronological holdout evaluation; no random shuffling."""
    group = group.sort_values("Date")
    y = group["Occupied Beds"].astype(float).reset_index(drop=True)
    if len(y) < 60:
        return {"MAE": float("nan"), "RMSE": float("nan"), "Test days": 0}
    split = max(30, len(y) - test_days)
    train = y.iloc[:split].reset_index(drop=True)
    actual = y.iloc[split:].reset_index(drop=True)
    history = train.tolist()
    predictions = []
    # Walk forward: each new prediction only uses earlier observations.
    for _ in range(len(actual)):
        s = pd.Series(history, dtype=float)
        X = _features(s)
        valid = X.notna().all(axis=1)
        model = Ridge(alpha=2.0)
        model.fit(X.loc[valid, FEATURES], s.loc[valid])
        next_idx = len(history)
        weekday = next_idx % 7
        next_features = pd.DataFrame([{
            "time_index": float(next_idx),
            "weekday_sin": np.sin(2*np.pi*weekday/7),
            "weekday_cos": np.cos(2*np.pi*weekday/7),
            "lag_1": history[-1],
            "lag_7": history[-7] if len(history) >= 7 else history[-1],
            "rolling_7": float(np.mean(history[-7:])),
        }])
        pred = max(0.0, float(model.predict(next_features[FEATURES])[0]))
        predictions.append(pred)
        history.append(float(actual.iloc[len(predictions)-1]))
    return {
        "MAE": float(mean_absolute_error(actual, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
        "Test days": len(actual),
    }

def forecast_department(group: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Fit on all available history and recursively forecast the requested horizon."""
    group = group.sort_values("Date")
    y = group["Occupied Beds"].astype(float).reset_index(drop=True)
    model = _fit_model(y)
    history = y.tolist()
    last_date = pd.to_datetime(group["Date"].max())
    rows = []
    capacity = int(group["Available Beds"].iloc[-1])
    for step in range(1, horizon + 1):
        idx = len(history)
        weekday = idx % 7
        features = pd.DataFrame([{
            "time_index": float(idx),
            "weekday_sin": np.sin(2*np.pi*weekday/7),
            "weekday_cos": np.cos(2*np.pi*weekday/7),
            "lag_1": history[-1],
            "lag_7": history[-7] if len(history) >= 7 else history[-1],
            "rolling_7": float(np.mean(history[-7:])),
        }])
        pred = max(0.0, float(model.predict(features[FEATURES])[0]))
        # Demand can exceed current capacity in a forecast; do not clip it to capacity.
        rows.append({
            "Date": last_date + pd.Timedelta(days=step),
            "Predicted Occupied Beds": pred,
            "Existing Capacity": capacity,
        })
        history.append(pred)
    return pd.DataFrame(rows)

def build_forecast(data: pd.DataFrame, departments, horizon: int, target_rate: float) -> pd.DataFrame:
    """Build forecast and capacity recommendations for selected departments."""
    pieces = []
    for dept in departments:
        group = data[data["Department"] == dept].sort_values("Date")
        if group.empty:
            continue
        fc = forecast_department(group, horizon)
        fc["Department"] = dept
        fc["Recommended Beds"] = np.ceil(fc["Predicted Occupied Beds"] / target_rate).astype(int)
        fc["Shortage / Surplus"] = fc["Existing Capacity"] - fc["Recommended Beds"]
        fc["Status"] = np.where(fc["Recommended Beds"] > fc["Existing Capacity"], "Potential shortage",
                                np.where(fc["Recommended Beds"] < fc["Existing Capacity"], "Capacity surplus",
                                         "At recommended capacity"))
        pieces.append(fc)
    if not pieces:
        return pd.DataFrame(columns=["Date", "Department", "Predicted Occupied Beds",
            "Existing Capacity", "Recommended Beds", "Shortage / Surplus", "Status"])
    out = pd.concat(pieces, ignore_index=True)
    return out[["Date", "Department", "Predicted Occupied Beds", "Existing Capacity",
                "Recommended Beds", "Shortage / Surplus", "Status"]]

def evaluate_all(data: pd.DataFrame, departments) -> pd.DataFrame:
    rows = []
    for dept in departments:
        group = data[data["Department"] == dept]
        metrics = evaluate_department(group)
        rows.append({"Department": dept, **metrics})
    return pd.DataFrame(rows)
