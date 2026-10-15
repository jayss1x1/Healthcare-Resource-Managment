"""Reusable metrics for the dashboard."""
import numpy as np
import pandas as pd

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Bed Occupancy Rate (%)"] = 100 * out["Occupied Beds"] / out["Available Beds"].replace(0, np.nan)
    out["Equipment Utilization (%)"] = 100 * out["Equipment in Use"] / out["Equipment Available"].replace(0, np.nan)
    out["Admissions per Staff"] = out["Patient Admissions"] / out["Staff Available"].replace(0, np.nan)
    out["Medicine Usage vs Stock (%)"] = 100 * out["Medicine Used Units"] / out["Medicine Stock Units"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)
