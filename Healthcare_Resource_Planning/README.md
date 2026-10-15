# Healthcare Resource Planning System Using Predictive Analysis

A beginner-friendly Data Science project that explores historical hospital resource usage and forecasts occupied beds for the next 7, 14, or 30 days.

> **Important:** The CSV is generated synthetic sample data. It is not real hospital data. The forecasts are educational estimates, not medically validated recommendations.

## Objectives

- Explore daily admissions, bed occupancy, medicine usage, staffing and equipment use.
- Forecast occupied beds, nurse demand, medicine usage and medical equipment demand for four sample departments.
- Recommend capacity at a configurable target occupancy rate (default 85%).
- Evaluate forecasting with a chronological holdout using MAE and RMSE.
- Present the analysis in an interactive Streamlit dashboard.

## Features and modules

- `generate_data.py`: creates 730 days of reproducible synthetic daily data per department.
- `src/data_loader.py`: reads the CSV, checks required columns, parses dates and applies basic consistency checks.
- `src/analysis.py`: calculates bed occupancy, equipment utilization, admissions per staff and medicine usage-to-stock ratio.
- `src/forecasting.py`: trains an explainable Ridge regression model, evaluates it chronologically and recursively forecasts future occupied beds.
- `src/resource_forecasting.py`: produces four-resource forecasts. Nurse requirements use a configurable illustrative beds-per-nurse ratio; medicine demand forecasts daily usage with a simple 10% buffer; equipment demand forecasts equipment in use and applies a target utilization.
- `app.py`: interactive dashboard with overview, historical charts, forecasts, utilization metrics, filters and CSV downloads.

## Technology

Python 3.11+, Pandas, NumPy, Matplotlib, Seaborn, Streamlit and Scikit-learn. Data is stored in CSV; no database server is needed.

## Dataset

`data/healthcare_resources.csv` includes:
- Date
- Department
- Patient Admissions
- Available Beds
- Occupied Beds
- Staff Available
- Medicine Stock Units
- Medicine Used Units
- Equipment Available
- Equipment in Use

The generator creates two years of daily records for General Medicine, Surgery, Pediatrics and Orthopedics. `Nurses Available` is a synthetic staffing count for this project. Randomness is seeded for reproducibility. Historical occupied beds are constrained not to exceed available beds. This is synthetic demonstration data and should not be interpreted as representing a real hospital.

## Four-resource prediction assumptions

- **Beds:** recommended capacity = ceiling(predicted occupied beds / target bed occupancy rate).
- **Nurses:** estimated nurses required = ceiling(predicted occupied beds / selected beds-per-nurse ratio). This is a classroom assumption and does not model shifts, leave, patient acuity or local staffing standards.
- **Medicines:** the model forecasts daily medicine units used. A simple daily recommendation adds a 10% buffer. It is not medicine-specific and does not model expiry dates, delivery lead time or clinical dosing.
- **Medical equipment:** the model forecasts equipment in use, then estimates required equipment as ceiling(predicted use / target equipment utilization). The synthetic dataset treats equipment as a single generic category.

These estimates are for demonstration only, not validated hospital procurement or staffing advice.

## Forecasting methodology

For each department, Ridge regression uses:
- A time index to represent gradual trends.
- Sine and cosine weekday features to model weekly patterns.
- Previous-day occupancy (`lag_1`).
- Previous-week occupancy (`lag_7`).
- The mean of the preceding seven observations (`rolling_7`).

Forecasts are recursive: each predicted value is appended to the history to produce the next day's features. Predictions are clamped at zero but are **not** clipped to existing capacity, so a potential shortage can be visible.

### Recommended capacity

`Recommended Beds = ceiling(Predicted Occupied Beds / Target Occupancy Rate)`

Example: if predicted occupied beds = 85 and target occupancy = 0.85, recommended beds = ceiling(85 / 0.85) = 100.

`Shortage / Surplus = Existing Capacity - Recommended Beds`

A negative value indicates that recommended capacity exceeds current capacity.

### Evaluation

The latest 30 historical days are held out chronologically. The evaluation walks forward day by day: each prediction uses only earlier observations; the actual value is then added to history before predicting the next day. This avoids random shuffling and future-to-past leakage.

- **MAE (Mean Absolute Error):** average absolute difference between prediction and actual occupied beds.
- **RMSE (Root Mean Squared Error):** square root of the average squared prediction error; larger errors are penalized more.

Scores are shown in the dashboard and should be interpreted as a historical check, not a guarantee of future performance.

## Graph guide

- **Daily occupied beds:** shows changes in total occupancy over time for selected departments.
- **Daily admissions:** shows changes in admission volume.
- **Occupied vs available beds:** compares average historical occupancy and capacity by department.
- **Medicine stock vs usage:** compares average recorded stock units with daily units used.
- **Equipment available vs in use:** compares average equipment inventory and active use.
- **Historical and predicted occupied beds:** shows recent history and forecast values with a forecast-start divider.
- **Predicted demand vs recommended/existing capacity:** helps spot dates where capacity may be insufficient.
- **Recommended beds by department:** compares forecast capacity needs.
- **Utilization table and chart:** compares occupancy, equipment usage, admissions-per-staff and medicine usage-to-stock ratios.

## Installation on Windows

1. Install Python 3.11 or later from the official Python website. During installation, enable **Add Python to PATH**.
2. Extract the ZIP folder.
3. Open Command Prompt or PowerShell in `Healthcare_Resource_Planning`.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or continue with the Python executable inside `.venv`.

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Generate the CSV:

   ```powershell
   python generate_data.py
   ```

7. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

8. Streamlit prints a local URL, normally `http://localhost:8501`. Open it in your browser.

## How to demonstrate

1. Show `data/healthcare_resources.csv` and explain that it is generated synthetic data.
2. In Overview, point out current available beds, occupied beds and occupancy percentage.
3. In Historical Analysis, explain the admissions and occupancy patterns and compare departments.
4. In Future Bed Prediction, switch between 7, 14 and 30 days and change the target occupancy rate.
5. Explain the recommended-bed formula and show the shortage/surplus column.
6. Point to the MAE/RMSE evaluation table and explain that smaller values are better, but they do not prove clinical accuracy.
7. Download the forecast CSV and explore/filter the underlying data.

## Basic tests

From the project folder, run:

```powershell
python generate_data.py
python -c "from src.data_loader import load_data; d=load_data(); print(d.shape); print((d['Occupied Beds'] <= d['Available Beds']).all())"
python -c "from src.data_loader import load_data; from src.forecasting import build_forecast, evaluate_all; d=load_data(); depts=sorted(d['Department'].unique()); print(build_forecast(d,depts,7,0.85).head()); print(evaluate_all(d,depts))"
```

The consistency test should print `True`. Forecast values and evaluation metrics are computed from the generated CSV rather than hardcoded into the dashboard.

## Limitations and future enhancements

- Synthetic data cannot capture real hospital seasonality, outbreaks, patient acuity or policy changes.
- The model is intentionally simple and recursive predictions can drift over longer horizons.
- Staffing, medicines and equipment are represented by basic quantities, not clinically approved planning rules.
- Real deployment would require validated, privacy-protected data, stronger backtesting, uncertainty intervals and review by hospital operations experts.
- Possible enhancements: compare multiple models, include holidays and external events, add confidence intervals, improve anomaly detection, and support securely uploaded real datasets.
