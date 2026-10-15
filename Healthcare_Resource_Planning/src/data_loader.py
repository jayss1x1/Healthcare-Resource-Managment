"""Load and validate the project's CSV dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "healthcare_resources.csv"
REQUIRED_COLUMNS = [
    "Date", "Department", "Patient Admissions", "Available Beds", "Occupied Beds",
    "Staff Available", "Nurses Available", "Medicine Stock Units", "Medicine Used Units",
    "Equipment Available", "Equipment in Use",
]

def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load CSV, validate columns, parse dates, and repair basic invalid values."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python generate_data.py` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Department"]).copy()
    numeric = [c for c in REQUIRED_COLUMNS if c not in ("Date", "Department")]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
    df["Available Beds"] = df["Available Beds"].clip(lower=1)
    # Keep the historical data internally consistent.
    df["Occupied Beds"] = df[["Occupied Beds", "Available Beds"]].min(axis=1)
    df["Equipment in Use"] = df[["Equipment in Use", "Equipment Available"]].min(axis=1)
    return df.sort_values(["Department", "Date"]).reset_index(drop=True)
