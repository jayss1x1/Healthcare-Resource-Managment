"""Generate a reproducible synthetic hospital resource dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "healthcare_resources.csv"

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
                          periods=730, freq="D")
    departments = {
        "General Medicine": {"beds": 100, "base": 67, "staff": 34, "admissions": 25},
        "Surgery": {"beds": 70, "base": 44, "staff": 27, "admissions": 15},
        "Pediatrics": {"beds": 50, "base": 29, "staff": 20, "admissions": 13},
        "Orthopedics": {"beds": 60, "base": 37, "staff": 22, "admissions": 11},
    }
    rows = []
    for dept, cfg in departments.items():
        phase = rng.uniform(0, 2 * np.pi)
        for i, date in enumerate(dates):
            weekday_effect = -3 if date.dayofweek >= 5 else 1
            annual_effect = 5 * np.sin(2 * np.pi * i / 365.25 + phase)
            trend = i * 0.004
            noise = rng.normal(0, 3.3)
            occupied = int(np.clip(round(cfg["base"] + weekday_effect + annual_effect + trend + noise),
                                   0, cfg["beds"]))
            admissions = max(0, int(round(cfg["admissions"] + weekday_effect * 0.7 +
                                          2.5 * np.sin(2 * np.pi * i / 30) + rng.normal(0, 3))))
            staff = max(1, int(round(cfg["staff"] + rng.normal(0, 2))))
            med_stock = max(0, int(round(cfg["beds"] * 7.5 + rng.normal(0, 35))))
            med_used = max(0, int(round(occupied * 1.7 + admissions * 0.8 + rng.normal(0, 8))))
            equipment = max(1, int(round(cfg["beds"] * 0.55)))
            equipment_use = int(np.clip(round(equipment * (occupied / cfg["beds"]) +
                                                rng.normal(0, 2)), 0, equipment))
            rows.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Department": dept,
                "Patient Admissions": admissions,
                "Available Beds": cfg["beds"],
                "Occupied Beds": occupied,
                "Staff Available": staff,
                "Nurses Available": staff,
                "Medicine Stock Units": med_stock,
                "Medicine Used Units": med_used,
                "Equipment Available": equipment,
                "Equipment in Use": equipment_use,
            })
    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df):,} synthetic daily department records: {DATA_PATH}")
    print("NOTE: These are synthetic educational data, not real hospital records.")
