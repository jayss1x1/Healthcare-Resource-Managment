"""Forecast beds, nurses, medicine consumption, and equipment utilization."""
import numpy as np
import pandas as pd
from .forecasting import forecast_department

def forecast_resource_series(group, value_column, horizon):
    """Forecast a non-negative resource time series using the same transparent model."""
    temp = group[["Date", "Occupied Beds", "Available Beds"]].copy()
    temp["Occupied Beds"] = pd.to_numeric(group[value_column], errors="coerce").fillna(0).clip(lower=0).values
    temp["Available Beds"] = pd.to_numeric(group[value_column], errors="coerce").fillna(1).clip(lower=1).values
    # Reuse the tested department forecasting function, swapping in the selected historical series.
    return forecast_department(temp, horizon).rename(
        columns={"Predicted Occupied Beds": "Predicted Value", "Existing Capacity": "Existing Value"}
    )

def build_all_resource_forecasts(data, departments, horizon, target_occupancy=0.85,
                                 beds_per_nurse=5.0, equipment_target_utilization=0.80):
    """
    Create daily forecasts for four resource categories.
    Nurse demand is an explicit planning assumption based on predicted occupied beds.
    Medicine and equipment use historical daily series with recursive forecasting.
    """
    outputs = []
    for dept in departments:
        group = data[data["Department"] == dept].sort_values("Date")
        if group.empty:
            continue

        # Beds: forecast occupied beds and recommended capacity.
        bed = forecast_department(group, horizon)
        bed["Department"] = dept
        bed["Resource"] = "Beds"
        bed["Predicted Demand"] = bed["Predicted Occupied Beds"]
        bed["Current Available"] = int(group["Available Beds"].iloc[-1])
        bed["Recommended Quantity"] = np.ceil(bed["Predicted Demand"] / target_occupancy).astype(int)
        bed["Shortage / Surplus"] = bed["Current Available"] - bed["Recommended Quantity"]
        outputs.append(bed[["Date", "Department", "Resource", "Predicted Demand", "Current Available",
                            "Recommended Quantity", "Shortage / Surplus"]])

        # Nurses: planning estimate derived from predicted occupied beds and an adjustable ratio.
        nurse = bed[["Date", "Department", "Predicted Occupied Beds"]].copy()
        nurse["Resource"] = "Nurses"
        nurse["Predicted Demand"] = nurse["Predicted Occupied Beds"] / beds_per_nurse
        nurse["Recommended Quantity"] = np.ceil(nurse["Predicted Demand"]).astype(int)
        nurse["Current Available"] = int(group["Nurses Available"].iloc[-1])
        nurse["Shortage / Surplus"] = nurse["Current Available"] - nurse["Recommended Quantity"]
        outputs.append(nurse[["Date", "Department", "Resource", "Predicted Demand", "Current Available",
                              "Recommended Quantity", "Shortage / Surplus"]])

        # Medicines: forecast daily units consumed; recommend projected horizon usage plus 10% buffer.
        med = forecast_resource_series(group, "Medicine Used Units", horizon)
        med["Department"] = dept
        med["Resource"] = "Medicines"
        med["Predicted Demand"] = med["Predicted Value"]
        med["Current Available"] = int(group["Medicine Stock Units"].iloc[-1])
        med["Recommended Quantity"] = np.ceil(med["Predicted Demand"] * 1.10).astype(int)
        med["Shortage / Surplus"] = med["Current Available"] - med["Recommended Quantity"]
        outputs.append(med[["Date", "Department", "Resource", "Predicted Demand", "Current Available",
                            "Recommended Quantity", "Shortage / Surplus"]])

        # Equipment: forecast daily equipment in use; recommend enough equipment at target utilization.
        equip = forecast_resource_series(group, "Equipment in Use", horizon)
        equip["Department"] = dept
        equip["Resource"] = "Medical Equipment"
        equip["Predicted Demand"] = equip["Predicted Value"]
        equip["Current Available"] = int(group["Equipment Available"].iloc[-1])
        equip["Recommended Quantity"] = np.ceil(equip["Predicted Demand"] / equipment_target_utilization).astype(int)
        equip["Shortage / Surplus"] = equip["Current Available"] - equip["Recommended Quantity"]
        outputs.append(equip[["Date", "Department", "Resource", "Predicted Demand", "Current Available",
                              "Recommended Quantity", "Shortage / Surplus"]])

    if not outputs:
        return pd.DataFrame(columns=["Date", "Department", "Resource", "Predicted Demand",
                                     "Current Available", "Recommended Quantity", "Shortage / Surplus"])
    result = pd.concat(outputs, ignore_index=True)
    result["Status"] = np.where(result["Shortage / Surplus"] < 0, "Potential shortage",
                                np.where(result["Shortage / Surplus"] > 0, "Capacity / stock surplus",
                                         "At recommended quantity"))
    return result.sort_values(["Date", "Resource", "Department"]).reset_index(drop=True)
