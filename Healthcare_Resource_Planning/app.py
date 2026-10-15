from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_data
from src.analysis import add_metrics
from src.forecasting import build_forecast, evaluate_all
from src.resource_forecasting import build_all_resource_forecasts

st.set_page_config(page_title="Healthcare Resource Planning", page_icon="🏥", layout="wide")
st.markdown("""
<style>
.stApp { background: #f4f7fb; color: #172033; }
section[data-testid="stMain"] { background: #f4f7fb; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #14213d !important; }
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"] { color: #263449 !important; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #d8e1ee; border-radius: 12px; padding: 14px; }
[data-testid="stSidebar"] { background: #202633; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f8fafc !important; }
button[data-baseweb="tab"] { color: #263449 !important; }
[data-testid="stDataFrame"] { border: 1px solid #d8e1ee; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Healthcare Resource Planning System")
st.caption("Hospital resource trends and demand forecasts • Upload a dataset or explore the included sample data")

from src.data_loader import REQUIRED_COLUMNS

@st.cache_data
def get_data():
    return load_data()

def validate_uploaded_csv(file_obj):
    """Read and validate a CSV uploaded by the user using the project's expected schema."""
    uploaded = pd.read_csv(file_obj)
    missing = [col for col in REQUIRED_COLUMNS if col not in uploaded.columns]
    if missing:
        raise ValueError(
            "This CSV is missing required columns: " + ", ".join(missing)
            + ". Download the CSV template below, fill it with your data, and upload it again."
        )
    uploaded["Date"] = pd.to_datetime(uploaded["Date"], errors="coerce")
    uploaded["Department"] = uploaded["Department"].astype("string").str.strip()
    uploaded = uploaded.dropna(subset=["Date", "Department"])
    uploaded = uploaded[uploaded["Department"] != ""].copy()
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("Date", "Department")]
    for col in numeric_cols:
        uploaded[col] = pd.to_numeric(uploaded[col], errors="coerce")
    if uploaded[numeric_cols].isna().any().any():
        bad = [c for c in numeric_cols if uploaded[c].isna().any()]
        raise ValueError("These columns contain blank or non-numeric values: " + ", ".join(bad))
    uploaded[numeric_cols] = uploaded[numeric_cols].clip(lower=0)
    uploaded["Available Beds"] = uploaded["Available Beds"].clip(lower=1)
    uploaded["Occupied Beds"] = uploaded[["Occupied Beds", "Available Beds"]].min(axis=1)
    uploaded["Equipment in Use"] = uploaded[["Equipment in Use", "Equipment Available"]].min(axis=1)
    if uploaded.empty:
        raise ValueError("No usable rows found. Check that Date and Department are filled in.")
    return uploaded.sort_values(["Department", "Date"]).reset_index(drop=True)

# Put data upload in the main page so it is easy to find.
st.subheader("1. Choose your data")
upload_col, template_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload hospital data (.csv)", type=["csv"], key="hospital_csv_upload",
        help="For forecasts, include the fields shown in the CSV template. Your file stays in this local app session."
    )
with template_col:
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    st.download_button(
        "Download CSV template", data=template.to_csv(index=False).encode("utf-8"),
        file_name="healthcare_resources_template.csv", mime="text/csv", use_container_width=True
    )
with st.expander("Which columns does my CSV need?", expanded=False):
    st.write("Use one row per date and department. Dates should be readable dates (for example, 2026-01-31); numeric fields should contain numbers only.")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.write("If your column names or data structure are different, this version will ask you to align the file to the template before forecasting. Each resource prediction needs historical values for that resource.")

try:
    if uploaded_file is not None:
        raw = validate_uploaded_csv(uploaded_file)
        st.success(f"Uploaded dataset loaded: {len(raw):,} rows · {raw['Department'].nunique()} departments · {raw['Date'].min():%d %b %Y} to {raw['Date'].max():%d %b %Y}")
        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
        data_source_label = "Uploaded CSV"
    else:
        raw = get_data()
        data_source_label = "Included sample data"
        st.info("No CSV uploaded. The dashboard is currently using the included sample dataset. Upload your own file above to replace it.")
except Exception as exc:
    st.error(f"The CSV could not be used: {exc}")
    st.warning("Download the template above, copy your records into it, and keep the column headers unchanged. Then upload the completed CSV.")
    st.stop()

df = add_metrics(raw)
departments = sorted(df["Department"].dropna().unique().tolist())
latest_date = df["Date"].max()
current = df[df["Date"] == latest_date].copy()
total_beds = int(current["Available Beds"].sum())
occupied = int(current["Occupied Beds"].sum())
occupancy = occupied / total_beds * 100 if total_beds else 0
admissions_total = int(current["Patient Admissions"].sum())

with st.sidebar:
    st.header("Dashboard controls")
    chosen_depts = st.multiselect("Departments", departments, default=departments)
    if not chosen_depts:
        chosen_depts = departments
    horizon_label = st.selectbox("Forecast horizon", ["7 days", "14 days", "30 days"], index=0)
    horizon = int(horizon_label.split()[0])
    target_rate = st.slider("Target occupancy rate", min_value=0.60, max_value=0.95, value=0.85, step=0.05,
                            help="Recommended beds = ceiling(predicted occupied beds ÷ target occupancy rate).")
    st.caption("Forecasts are estimates. Review assumptions and validate results before making operational decisions.")
    st.markdown("**Resource planning assumptions**")
    beds_per_nurse = st.number_input("Occupied beds per nurse (illustrative)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    equipment_target = st.slider("Target equipment utilization", min_value=0.50, max_value=0.95, value=0.80, step=0.05)

selected = df[df["Department"].isin(chosen_depts)].copy()
current_sel = current[current["Department"].isin(chosen_depts)]
try:
    forecast = build_forecast(raw, chosen_depts, horizon, target_rate)
except Exception as exc:
    st.error(f"Forecast could not be calculated: {exc}")
    st.stop()

risk_departments = forecast.loc[forecast["Recommended Beds"] > forecast["Existing Capacity"], "Department"].nunique()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Available beds", f"{int(current_sel['Available Beds'].sum()):,}")
c2.metric("Occupied beds", f"{int(current_sel['Occupied Beds'].sum()):,}")
bed_total = current_sel["Available Beds"].sum()
c3.metric("Occupancy", f"{(current_sel['Occupied Beds'].sum()/bed_total*100 if bed_total else 0):.1f}%")
c4.metric("Admissions (latest day)", f"{int(current_sel['Patient Admissions'].sum()):,}")
c5.metric("Departments at risk", f"{risk_departments}")

tabs = st.tabs(["Overview", "Historical Analysis", "Future Bed Prediction", "Four-Resource Prediction", "Resource Utilization", "Data Explorer"])

with tabs[0]:
    st.subheader("Current snapshot")
    st.caption(f"Data source: {data_source_label}. The summary uses the latest date in the selected dataset.")
    st.write(f"Latest date in dataset: **{latest_date:%d %b %Y}**. Summary cards use the latest available daily record for each selected department.")
    col1, col2 = st.columns(2)
    dept_now = current_sel.groupby("Department", as_index=False)[["Occupied Beds", "Available Beds"]].sum()
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(dept_now))
        ax.bar(x - .18, dept_now["Occupied Beds"], .36, label="Occupied", color="#168aad")
        ax.bar(x + .18, dept_now["Available Beds"], .36, label="Available", color="#b7e4f5")
        ax.set_xticks(x, dept_now["Department"], rotation=20, ha="right")
        ax.set_ylabel("Number of beds"); ax.set_title("Beds on the latest recorded date"); ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col2:
        status = forecast.groupby("Department")["Status"].apply(lambda s: (s == "Potential shortage").any()).reset_index(name="At risk")
        st.dataframe(status, use_container_width=True, hide_index=True)
        st.info("A department is flagged if at least one forecast date requires more beds than its existing capacity.")

with tabs[1]:
    st.subheader("Historical data analysis")
    hist = selected.groupby("Date", as_index=False)[["Occupied Beds", "Patient Admissions"]].sum()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist["Date"], hist["Occupied Beds"], color="#087e8b", linewidth=1.8)
        ax.set_title("Daily occupied beds"); ax.set_xlabel("Date"); ax.set_ylabel("Occupied beds")
        fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist["Date"], hist["Patient Admissions"], color="#2563eb", linewidth=1.5)
        ax.set_title("Daily patient admissions"); ax.set_xlabel("Date"); ax.set_ylabel("Admissions")
        fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    dept_summary = selected.groupby("Department", as_index=False)[["Occupied Beds", "Available Beds"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=dept_summary.melt(id_vars="Department", var_name="Measure", value_name="Beds"),
                x="Department", y="Beds", hue="Measure", ax=ax, palette=["#0f766e", "#93c5fd"])
    ax.set_title("Average occupied and available beds by department"); ax.set_ylabel("Average number of beds")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    col3, col4 = st.columns(2)
    with col3:
        med = selected.groupby("Department", as_index=False)[["Medicine Stock Units", "Medicine Used Units"]].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        medm = med.melt(id_vars="Department", var_name="Measure", value_name="Units")
        sns.barplot(data=medm, x="Department", y="Units", hue="Measure", ax=ax)
        ax.set_title("Average medicine stock vs daily usage"); ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col4:
        equip = selected.groupby("Department", as_index=False)[["Equipment Available", "Equipment in Use"]].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        eqm = equip.melt(id_vars="Department", var_name="Measure", value_name="Units")
        sns.barplot(data=eqm, x="Department", y="Units", hue="Measure", ax=ax)
        ax.set_title("Average equipment available vs in use"); ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("These charts summarize daily records in the currently selected dataset. Use the chart title, horizontal axis (date or department), vertical axis (measured quantity), and legend to interpret each series.")

with tabs[2]:
    st.subheader("Future bed prediction")
    st.write("Forecasts are generated immediately using a Ridge regression model with time, weekday cycle, previous-day occupancy, previous-week occupancy, and a rolling 7-day mean.")
    evaluation = evaluate_all(raw, chosen_depts)
    st.markdown("**Held-out chronological evaluation** — the latest 30 days are evaluated in sequence, using only earlier observations to make each prediction.")
    st.dataframe(evaluation.round(2), use_container_width=True, hide_index=True)
    st.caption("MAE = mean absolute error in beds. RMSE = root mean squared error in beds. Lower values are better, but scores do not guarantee future accuracy.")
    st.divider()
    fc = forecast.copy()
    fc["Date"] = pd.to_datetime(fc["Date"])
    latest = selected["Date"].max()
    hist_line = selected.groupby("Date", as_index=False)["Occupied Beds"].sum()
    fc_line = fc.groupby("Date", as_index=False)["Predicted Occupied Beds"].sum()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        recent_hist = hist_line.tail(60)
        ax.plot(recent_hist["Date"], recent_hist["Occupied Beds"], label="Historical occupied", color="#087e8b")
        ax.plot(fc_line["Date"], fc_line["Predicted Occupied Beds"], label="Predicted occupied", color="#e07a5f", linestyle="--", marker="o")
        ax.axvline(pd.to_datetime(latest), color="#64748b", linestyle=":", label="Forecast starts")
        ax.set_title("Historical and predicted occupied beds"); ax.set_xlabel("Date"); ax.set_ylabel("Beds")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col2:
        total_fc = fc.groupby("Date", as_index=False).agg(
            Predicted=("Predicted Occupied Beds", "sum"),
            Recommended=("Recommended Beds", "sum"),
            Existing=("Existing Capacity", "sum"))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(total_fc["Date"], total_fc["Predicted"], label="Predicted demand", marker="o", color="#e07a5f")
        ax.plot(total_fc["Date"], total_fc["Recommended"], label="Recommended capacity", color="#0f766e")
        ax.plot(total_fc["Date"], total_fc["Existing"], label="Existing capacity", color="#2563eb", linestyle=":")
        ax.set_title("Predicted demand and bed capacity"); ax.set_xlabel("Date"); ax.set_ylabel("Beds")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    dept_rec = fc.groupby("Department", as_index=False)["Recommended Beds"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=dept_rec, x="Department", y="Recommended Beds", ax=ax, color="#2a9d8f")
    ax.set_title("Average recommended beds by department"); ax.set_ylabel("Average beds")
    ax.tick_params(axis="x", rotation=15); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    display_fc = fc.copy()
    display_fc["Predicted Occupied Beds"] = display_fc["Predicted Occupied Beds"].round(1)
    st.markdown("**Forecast table**")
    st.dataframe(display_fc, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download forecast CSV", data=fc.to_csv(index=False).encode("utf-8"),
                       file_name=f"bed_forecast_{horizon}days.csv", mime="text/csv")
    shortages = fc[fc["Recommended Beds"] > fc["Existing Capacity"]]
    if not shortages.empty:
        st.warning(f"Potential shortage flagged on {shortages['Date'].nunique()} forecast date(s) across the selected departments.")
    else:
        st.success("No capacity shortage flagged for this forecast horizon under the selected target occupancy rate.")
    st.caption("Recommended Beds = ceiling(predicted occupied beds ÷ target occupancy rate). Shortage / Surplus = existing capacity − recommended beds; a negative number indicates a shortage.")


with tabs[3]:
    st.subheader("Four-resource prediction")
    st.write("Forecast bed demand, nursing demand, medicine usage, and medical equipment demand from the selected dataset.")
    st.info("How these estimates are calculated: nurse demand uses occupied beds divided by the selected beds-per-nurse ratio; medicine recommendation adds a 10% buffer to predicted daily usage; equipment recommendation accounts for the target utilization. Adjust these assumptions to match your organization and have domain experts review them.")
    all_resources = build_all_resource_forecasts(
        raw, chosen_depts, horizon, target_occupancy=target_rate,
        beds_per_nurse=beds_per_nurse, equipment_target_utilization=equipment_target
    )
    resource_names = ["Beds", "Nurses", "Medicines", "Medical Equipment"]
    resource_choice = st.selectbox("Choose resource to inspect", resource_names)
    resource_view = all_resources[all_resources["Resource"] == resource_choice].copy()
    if resource_view.empty:
        st.warning("No forecast rows are available for this selection.")
    else:
        totals = resource_view.groupby("Date", as_index=False).agg(
            Predicted=("Predicted Demand", "sum"),
            Recommended=("Recommended Quantity", "sum"),
            Current=("Current Available", "sum")
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(totals["Date"], totals["Predicted"], label="Predicted demand", marker="o", color="#e07a5f")
        ax.plot(totals["Date"], totals["Recommended"], label="Recommended quantity", color="#0f766e")
        ax.plot(totals["Date"], totals["Current"], label="Current available", linestyle=":", color="#2563eb")
        ax.set_title(f"{resource_choice}: forecast demand and resource availability")
        ax.set_xlabel("Date"); ax.set_ylabel("Quantity"); ax.legend()
        fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        dept_totals = resource_view.groupby("Department", as_index=False).agg(
            Predicted=("Predicted Demand", "mean"),
            Recommended=("Recommended Quantity", "mean"),
            Current=("Current Available", "mean")
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        dept_long = dept_totals.melt(id_vars="Department", var_name="Measure", value_name="Quantity")
        sns.barplot(data=dept_long, x="Department", y="Quantity", hue="Measure", ax=ax)
        ax.set_title(f"Average {resource_choice.lower()} quantities by department")
        ax.set_ylabel("Average quantity"); ax.tick_params(axis="x", rotation=15)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        shown = resource_view.copy()
        shown["Predicted Demand"] = shown["Predicted Demand"].round(2)
        st.markdown("**Resource forecast table**")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button(
            f"⬇️ Download {resource_choice.lower().replace(' ', '_')}_forecast.csv",
            data=resource_view.to_csv(index=False).encode("utf-8"),
            file_name=f"{resource_choice.lower().replace(' ', '_')}_forecast.csv",
            mime="text/csv"
        )
        risk = resource_view[resource_view["Shortage / Surplus"] < 0]
        if not risk.empty:
            st.warning(f"Potential shortage flagged on {risk['Date'].nunique()} date(s) for {resource_choice}.")
        else:
            st.success(f"No shortage flagged for {resource_choice} under the selected planning assumptions.")
    st.markdown("**How to interpret the four forecasts**")
    st.markdown("""
    - **Beds:** predicted occupied beds are converted into recommended capacity using the target occupancy rate.
    - **Nurses:** estimated nurses needed = ceiling(predicted occupied beds ÷ beds per nurse). Actual shift coverage and patient acuity are not modelled.
    - **Medicines:** predicted medicine usage is shown per day; recommended quantity is a simple 10% buffer over that day's predicted usage. Inventory expiry, delivery lead time and medicine-specific dosing are not modelled.
    - **Medical equipment:** predicted equipment in use is converted to recommended quantity using the target equipment utilization. Different equipment types are not modelled separately in this sample.
    """)

with tabs[4]:
    st.subheader("Resource utilization")
    st.markdown("""
    - **Bed occupancy rate** = occupied beds ÷ available beds × 100.
    - **Equipment utilization** = equipment in use ÷ equipment available × 100.
    - **Admissions per staff member** = patient admissions ÷ staff available; this is a simple workload indicator, not a staffing standard.
    - **Medicine usage vs stock** = medicine units used ÷ stock units × 100. It is a ratio, not a clinical inventory rule.
    """)
    metrics = selected.groupby("Department", as_index=False).agg(
        **{"Average bed occupancy (%)": ("Bed Occupancy Rate (%)", "mean"),
           "Average equipment utilization (%)": ("Equipment Utilization (%)", "mean"),
           "Average admissions per staff": ("Admissions per Staff", "mean"),
           "Average medicine usage vs stock (%)": ("Medicine Usage vs Stock (%)", "mean")})
    st.dataframe(metrics.round(2), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=metrics, x="Department", y="Average bed occupancy (%)", ax=ax, color="#168aad")
    ax.axhline(85, color="#e07a5f", linestyle="--", label="Illustrative 85% target")
    ax.set_ylim(0, max(100, metrics["Average bed occupancy (%)"].max() + 5))
    ax.set_title("Average bed occupancy by department"); ax.set_ylabel("Occupancy (%)"); ax.legend(); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.info("Metrics help compare patterns, but staffing and inventory decisions require local operational context and expert review.")

with tabs[5]:
    st.subheader("Data explorer")
    min_date, max_date = raw["Date"].min().date(), raw["Date"].max().date()
    date_range = st.date_input("Filter date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    explorer_depts = st.multiselect("Filter departments", departments, default=chosen_depts, key="explorer_depts")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = raw[(raw["Date"] >= start_date) & (raw["Date"] <= end_date)]
    else:
        filtered = raw[raw["Date"].dt.date == date_range]
    if explorer_depts:
        filtered = filtered[filtered["Department"].isin(explorer_depts)]
    st.write(f"Rows: **{len(filtered):,}**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download filtered data CSV", data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_healthcare_resources.csv", mime="text/csv")

st.divider()
st.caption("Forecasts are decision-support estimates, not a substitute for hospital policies, clinical judgment, or validated operational planning.")
