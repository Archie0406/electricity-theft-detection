"""
data_preprocessing.py
----------------------
Handles all data cleaning steps BEFORE any feature engineering or
model training happens. Keeping preprocessing separate from feature
engineering and modeling avoids data leakage and makes the pipeline
easier to explain in an interview.

Two imputation paths are provided on purpose:

    1. clean_dataset() - a convenience, ALL-DATA cleaning function used
       for quick exploration (e.g. in notebooks/exploratory_analysis.ipynb)
       where there is no train/test split yet and the goal is just to
       look at the data.

    2. fit_missing_value_stats() / apply_missing_value_stats() - the
       LEAK-SAFE version used by the actual training pipeline
       (src/train_model.py). Statistics (medians) are computed using
       ONLY the training portion of the data, then reused to fill the
       test portion, exactly like a scikit-learn Pipeline's fit/transform
       would work with a real Imputer object.

time_based_split() implements the train/test split itself: earlier
months are used for training, later months for testing, which mirrors
how the system would actually be used - trained on historical data,
then applied to new/unseen consumption records.

Steps performed by the full pipeline:
    1. Load raw data
    2. Remove duplicate rows
    3. Fix data types
    4. Time-based train/test split
    5. Fit missing-value statistics on TRAIN only, apply to both
    6. Detect (but not blindly remove) outliers
    7. Return clean train/test DataFrames ready for feature engineering
"""

import pandas as pd
import numpy as np


# Dataset covers 6 months per customer (see src/generate_dataset.py).
# Earlier months simulate "historical" data the model would have been
# trained on; later months simulate "new" consumption data the model
# has never seen - the real-world scenario this project simulates.
TRAIN_MONTHS = [1, 2, 3, 4]
TEST_MONTHS = [5, 6]

NUMERIC_COLUMNS = [
    "Electricity_Consumption_kWh",
    "Previous_Month_Consumption",
    "Average_Consumption_3_Months",
    "Average_Consumption_6_Months",
    "Peak_Consumption",
    "Off_Peak_Consumption",
    "Voltage",
    "Current",
    "Power_Factor",
    "Meter_Age",
    "Number_of_Meter_Readings",
]

CATEGORICAL_COLUMNS = ["Location_Type", "Customer_Type"]


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (common issue in real utility data exports)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[preprocessing] Removed {removed} duplicate rows")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric and categorical columns are strings."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    ALL-DATA missing-value handling, for exploration/EDA only.

    Strategy:
    - Voltage, Power_Factor, Current: fill with the MEDIAN of the same
      Customer_Type group. Median is used instead of mean because it is
      robust to outliers (and consumption data is often skewed).
    - Previous_Month_Consumption: for a customer's first recorded month
      this is naturally missing (there is no previous month) - fill with
      the current month's consumption as a reasonable default.

    NOTE: This computes medians across the WHOLE dataset, which is fine
    for a quick look at the data but is NOT used by the actual training
    pipeline (train_model.py uses fit_missing_value_stats /
    apply_missing_value_stats instead, so that only TRAIN-month data is
    used to compute the fill values - see the module docstring above).
    """
    df = df.copy()

    missing_before = df[NUMERIC_COLUMNS].isna().sum().sum()

    for col in ["Voltage", "Power_Factor", "Current"]:
        if col in df.columns:
            df[col] = df.groupby("Customer_Type")[col].transform(
                lambda s: s.fillna(s.median())
            )
            # fallback in case an entire group was NaN
            df[col] = df[col].fillna(df[col].median())

    if "Previous_Month_Consumption" in df.columns:
        df["Previous_Month_Consumption"] = df["Previous_Month_Consumption"].fillna(
            df["Electricity_Consumption_kWh"]
        )

    missing_after = df[NUMERIC_COLUMNS].isna().sum().sum()
    print(f"[preprocessing] Missing values: {missing_before} -> {missing_after}")

    return df


def time_based_split(df: pd.DataFrame, train_months=None, test_months=None):
    """
    Split the dataset by MONTH rather than randomly by row.

    Random row-level splitting would let a customer's LATER months end
    up in the training set while an EARLIER month for the same customer
    ends up in the test set - effectively letting the model "see the
    future" relative to what it's being tested on. A time-based split
    avoids this: the model only ever trains on earlier consumption and
    is evaluated on later, genuinely unseen months - exactly how it
    would be used in production (train on history, score new data).

    Default: months 1-4 = train, months 5-6 = test (see TRAIN_MONTHS /
    TEST_MONTHS at the top of this file).
    """
    train_months = train_months or TRAIN_MONTHS
    test_months = test_months or TEST_MONTHS

    train_df = df[df["Month"].isin(train_months)].copy()
    test_df = df[df["Month"].isin(test_months)].copy()

    print(
        f"[split] Train months {train_months}: {len(train_df)} rows | "
        f"Test months {test_months}: {len(test_df)} rows"
    )
    return train_df, test_df


def fit_missing_value_stats(train_df: pd.DataFrame) -> dict:
    """
    Compute imputation statistics (per-Customer_Type medians) using
    ONLY the training portion of the data. This is the leak-safe
    equivalent of calling `.fit()` on a scikit-learn Imputer.
    """
    stats = {}
    for col in ["Voltage", "Power_Factor", "Current"]:
        if col in train_df.columns:
            stats[col] = train_df.groupby("Customer_Type")[col].median().to_dict()
            stats[f"{col}_global_fallback"] = float(train_df[col].median())
    return stats


def apply_missing_value_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Fill missing values using statistics computed earlier by
    fit_missing_value_stats(). This is the leak-safe equivalent of
    calling `.transform()` on a scikit-learn Imputer - the same
    train-derived values are applied to both train and test data.
    """
    df = df.copy()
    missing_before = df[["Voltage", "Power_Factor", "Current"]].isna().sum().sum()

    for col in ["Voltage", "Power_Factor", "Current"]:
        if col in df.columns and col in stats:
            group_medians = stats[col]
            fallback = stats.get(f"{col}_global_fallback", np.nan)
            mapped = df["Customer_Type"].map(group_medians)
            df[col] = df[col].fillna(mapped).fillna(fallback)

    if "Previous_Month_Consumption" in df.columns:
        df["Previous_Month_Consumption"] = df["Previous_Month_Consumption"].fillna(
            df["Electricity_Consumption_kWh"]
        )

    missing_after = df[["Voltage", "Power_Factor", "Current"]].isna().sum().sum()
    print(f"[preprocessing] Missing values in this split: {missing_before} -> {missing_after}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    Returns a boolean Series marking rows considered outliers.

    NOTE: We do NOT automatically remove outliers here. In electricity
    theft detection, outliers are often exactly the customers we are
    interested in investigating, so removing them would defeat the
    purpose of the project. We only FLAG them for awareness/EDA.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def outlier_summary(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Produce a small summary table of outlier counts per numeric column."""
    if columns is None:
        columns = ["Electricity_Consumption_kWh", "Peak_Consumption", "Power_Factor", "Current"]

    summary = []
    for col in columns:
        if col in df.columns:
            mask = detect_outliers_iqr(df, col)
            summary.append({"column": col, "outlier_count": int(mask.sum()), "outlier_pct": round(100 * mask.mean(), 2)})
    return pd.DataFrame(summary)


def clean_dataset(path: str) -> pd.DataFrame:
    """Full preprocessing pipeline: load -> dedupe -> fix types -> handle missing."""
    df = load_data(path)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = handle_missing_values(df)
    return df


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity_data.csv")
    cleaned = clean_dataset(data_path)
    print(cleaned.info())
    print("\nOutlier summary:")
    print(outlier_summary(cleaned))
