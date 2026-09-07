"""
prediction.py
---------------
Helper functions used by the Streamlit dashboard (and reusable
elsewhere) to load trained models and score customers.

Keeping this logic in its own module means the Streamlit app file
(app/app.py) stays focused on UI code, while all the "real" ML logic
lives here and in the other src/ modules - making the project easier
to navigate and explain.
"""

import os
import joblib
import numpy as np
import pandas as pd


def load_artifacts(models_dir: str):
    """
    Load all trained models and preprocessing/normalization objects.

    score_params (score_min / score_max) were computed from TRAINING
    months only (see train_model.py: fit_score_normalizer) and must be
    reused as-is here - never recalculated from whatever data is being
    scored - so that a new customer scored later gets the same 0-1
    anomaly scale as the training run.
    """
    iso_model = joblib.load(os.path.join(models_dir, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_list = joblib.load(os.path.join(models_dir, "feature_list.pkl"))
    score_params = joblib.load(os.path.join(models_dir, "anomaly_score_params.pkl"))
    return iso_model, scaler, feature_list, score_params


def _fmt(value, suffix="", decimals=1):
    """Format a numeric value for display, handling NaN (e.g. a customer's
    first month, which has no historical average yet) without crashing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A (no prior history)"
    return f"{value:.{decimals}f}{suffix}"


def load_scored_customers(data_path: str) -> pd.DataFrame:
    """Load the pre-scored dataset produced by train_model.py."""
    return pd.read_csv(data_path)


def get_customer_summary(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """Return all historical rows for a given customer, sorted by month."""
    subset = df[df["Customer_ID"] == customer_id].sort_values("Month")
    return subset


def get_latest_record(df: pd.DataFrame, customer_id: str) -> pd.Series:
    """Return the most recent monthly record for a customer."""
    subset = get_customer_summary(df, customer_id)
    if subset.empty:
        return None
    return subset.iloc[-1]


def format_customer_report(row: pd.Series) -> str:
    """Produce the plain-text style report shown in the dashboard/spec."""
    lines = [
        f"Customer ID: {row['Customer_ID']}",
        "",
        f"Risk Level: {row['Risk_Level'].upper()}",
        "",
        f"Previous 6-Month Average Consumption: {_fmt(row['Average_Consumption_6_Months'], ' kWh')}",
        f"Current Consumption: {_fmt(row['Electricity_Consumption_kWh'], ' kWh')}",
        f"Consumption Change (vs previous 3-month average): {_fmt(row['Consumption_Change_Pct'], '%')}",
        f"Anomaly Score: {_fmt(row['Anomaly_Score'], decimals=2)}",
        "",
        "Potential Reasons:",
    ]
    for reason in row["Flag_Reasons"].split(";"):
        lines.append(f"  - {reason.strip()}")

    if row["Risk_Level"] == "Normal":
        note = "This customer's consumption pattern currently looks typical based on available data."
    else:
        note = (
            "This result indicates an anomalous pattern and does not prove electricity theft.\n"
            "Further human investigation is required before any action is taken."
        )
    lines += ["", "Note:", note]
    return "\n".join(lines)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "scored_customers.csv")

    df = load_scored_customers(data_path)
    sample_id = df.iloc[0]["Customer_ID"]
    latest = get_latest_record(df, sample_id)
    print(format_customer_report(latest))
