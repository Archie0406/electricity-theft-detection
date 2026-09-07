"""
feature_engineering.py
------------------------
Creates the engineered features used by the anomaly-detection models.

Each feature is designed to capture a specific behavioral signal that
is relevant to electricity theft / meter tampering:

1. Consumption_Change_Pct
   How much current consumption differs from the customer's own recent
   HISTORICAL average (Average_Consumption_3_Months, which is computed
   in generate_dataset.py using only PAST months - see that file's
   docstring). A large negative change is the classic "sudden drop"
   signature associated with tampering. A customer's first month has no
   historical average yet (NaN) - Consumption_Change_Pct is set to 0 in
   that case, an explicit "no signal available yet" default rather than
   a value that would silently compare a month to itself.

2. Consumption_Deviation
   Absolute difference between current consumption and the HISTORICAL
   6-month average (Average_Consumption_6_Months, past months only), in
   kWh. Captures deviation regardless of direction. Also defaults to 0
   for a customer's first month, for the same reason as above.

3. Peak_OffPeak_Ratio
   Peak consumption divided by off-peak consumption. Legitimate
   customers usually have a fairly stable ratio; tampering that only
   affects certain hours can distort this ratio.

4. Consumption_Variability
   Coefficient of variation (std / mean) of the customer's own PAST
   consumption, computed over a rolling historical window (previous
   3 months by default). The current month's own consumption is
   intentionally excluded from this calculation, so the feature
   describes how stable/erratic the customer's history has been
   independent of whatever is happening this month - see
   add_consumption_variability() below for a worked example.

5. Reading_Completeness_Ratio
   Number of meter readings this month divided by the expected number
   of days (~30). Missed readings can indicate meter access/tampering
   issues.

6. MoM_Change_Pct
   Month-over-month % change using Previous_Month_Consumption (the
   customer's actual prior month). Deliberately separate from
   Consumption_Change_Pct (which uses a smoother 3-month average) - a
   sharp single-month drop and a gradual multi-month decline are
   different signals, and this feature exists specifically to make
   sure Previous_Month_Consumption (already present in the raw
   dataset) is put to genuine use rather than left unused. See
   add_mom_change_pct() below.

All engineered features are computed using ONLY information available
up to and including the current month for that customer (no future
information is used), which avoids data leakage. The one exception
that requires special care is Peak_OffPeak_Ratio's fallback value for
undefined ratios (division by zero) - see add_peak_offpeak_ratio() and
compute_peak_offpeak_fallback() below, and how train_model.py computes
that fallback from TRAINING rows only.
"""

import pandas as pd
import numpy as np


def add_consumption_change_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Avoid division by zero using a small epsilon
    denom = df["Average_Consumption_3_Months"].replace(0, np.nan)
    df["Consumption_Change_Pct"] = (
        (df["Electricity_Consumption_kWh"] - denom) / denom
    ) * 100
    df["Consumption_Change_Pct"] = df["Consumption_Change_Pct"].fillna(0)
    return df


def add_consumption_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Consumption_Deviation"] = (
        df["Electricity_Consumption_kWh"] - df["Average_Consumption_6_Months"]
    ).abs()
    # A customer's first month has no historical average (NaN) - default to
    # 0 deviation (no evidence of anomaly yet) rather than leaving NaN,
    # which would otherwise break scaling/model input downstream.
    df["Consumption_Deviation"] = df["Consumption_Deviation"].fillna(0)
    return df


def add_peak_offpeak_ratio(df: pd.DataFrame, fallback_value: float = None) -> pd.DataFrame:
    """
    Peak_OffPeak_Ratio can be undefined (division by zero) when a
    customer had zero off-peak consumption in a given month. Those gaps
    are filled with `fallback_value`.

    IMPORTANT (leakage): `fallback_value` should be computed from
    TRAINING rows only and passed in - see
    compute_peak_offpeak_fallback() below and its use in
    train_model.py. If no fallback is supplied (fallback_value=None),
    this falls back to computing the median from whatever DataFrame was
    passed in, which is only safe for all-data exploration/EDA (e.g.
    notebooks/exploratory_analysis.ipynb via clean_dataset()) where
    there is no train/test split yet. The actual training pipeline
    always supplies a train-only fallback so that no test-month
    statistic leaks into the value used to fill training (or test) rows.
    """
    df = df.copy()
    denom = df["Off_Peak_Consumption"].replace(0, np.nan)
    df["Peak_OffPeak_Ratio"] = df["Peak_Consumption"] / denom

    if fallback_value is None:
        fallback_value = df["Peak_OffPeak_Ratio"].median()

    df["Peak_OffPeak_Ratio"] = df["Peak_OffPeak_Ratio"].fillna(fallback_value)
    return df


def compute_peak_offpeak_fallback(df: pd.DataFrame) -> float:
    """
    Compute the Peak_OffPeak_Ratio fallback value from a given
    DataFrame. Call this on TRAINING rows ONLY (see train_model.py),
    then pass the result into add_peak_offpeak_ratio() / engineer_features()
    so the exact same train-derived fallback is reused for both training
    and test rows - never recomputed from a combined train+test dataset.
    """
    denom = df["Off_Peak_Consumption"].replace(0, np.nan)
    ratio = df["Peak_Consumption"] / denom
    return float(ratio.median())


def add_mom_change_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month-over-month consumption change, using Previous_Month_Consumption
    (the customer's actual prior month, generated directly in
    generate_dataset.py - genuinely past-only, never the current or a
    future month).

    This is DELIBERATELY separate from Consumption_Change_Pct (which
    compares against the smoother 3-month average): a single sharp
    month-to-month drop (this feature) is a different, complementary
    signal from a gradual drift away from a longer baseline (that
    feature). A customer whose consumption fell sharply just this one
    month, but whose 3-month average hasn't caught up yet, would be
    caught by this feature and missed by the other - and vice versa
    for a slow, gradual decline.

    A customer's first month has no previous month; that row's
    Previous_Month_Consumption is filled with its OWN current
    consumption during preprocessing (see data_preprocessing.py), which
    makes this formula naturally evaluate to 0% change - the same "no
    signal yet" default used by the other historical features.
    """
    df = df.copy()
    denom = df["Previous_Month_Consumption"].replace(0, np.nan)
    df["MoM_Change_Pct"] = ((df["Electricity_Consumption_kWh"] - denom) / denom) * 100
    df["MoM_Change_Pct"] = df["MoM_Change_Pct"].fillna(0)
    return df


def add_consumption_variability(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Consumption variability = coefficient of variation (std / mean) of a
    customer's own consumption over the `window` months immediately
    BEFORE the current row. The current month's own value is excluded
    on purpose - this is a HISTORICAL stability measure, not something
    a customer's own current-month reading can influence, which avoids
    the feature "explaining itself" and avoids using any future data.

    Example (window=3):
        Jan=280, Feb=290, Mar=275, Apr=285, May=100

        May's variability is computed from Feb, Mar, Apr only (the
        three months immediately before May). June's consumption is
        NEVER used when computing a feature for May - that would be
        future information the model wouldn't actually have at
        prediction time.

    Customers with fewer than 2 prior months of history (e.g. a
    customer's first or second recorded month) don't have enough data
    for a meaningful variability estimate. These rows are given a
    value of 0 (no evidence of instability yet) - an explicit,
    documented default rather than a silently guessed value.
    """
    df = df.copy()
    sorted_df = df.sort_values(["Customer_ID", "Month"])

    def _historical_cv(consumption_series: pd.Series) -> pd.Series:
        past_only = consumption_series.shift(1)  # exclude the current month
        rolling_std = past_only.rolling(window=window, min_periods=2).std()
        rolling_mean = past_only.rolling(window=window, min_periods=2).mean()
        return rolling_std / rolling_mean.replace(0, np.nan)

    sorted_df["Consumption_Variability"] = sorted_df.groupby("Customer_ID")[
        "Electricity_Consumption_kWh"
    ].transform(_historical_cv)
    sorted_df["Consumption_Variability"] = sorted_df["Consumption_Variability"].fillna(0)

    # Write the computed values back in the original row order.
    df["Consumption_Variability"] = sorted_df["Consumption_Variability"].reindex(df.index)
    return df


def add_reading_completeness_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    expected_readings = 30
    df["Reading_Completeness_Ratio"] = df["Number_of_Meter_Readings"] / expected_readings
    return df


def engineer_features(df: pd.DataFrame, peak_offpeak_fallback: float = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    peak_offpeak_fallback: the TRAIN-only median used to fill undefined
    Peak_OffPeak_Ratio values (see compute_peak_offpeak_fallback() and
    add_peak_offpeak_ratio()). Leave as None only for all-data
    exploration (e.g. notebooks) where there is no train/test split.
    """
    df = add_consumption_change_pct(df)
    df = add_consumption_deviation(df)
    df = add_mom_change_pct(df)
    df = add_peak_offpeak_ratio(df, fallback_value=peak_offpeak_fallback)
    df = add_consumption_variability(df)
    df = add_reading_completeness_ratio(df)
    return df


# Final feature list used for model training (numeric only - categorical
# columns are one-hot encoded separately in train_model.py)
ENGINEERED_FEATURES = [
    "Electricity_Consumption_kWh",
    "Consumption_Change_Pct",
    "MoM_Change_Pct",
    "Consumption_Deviation",
    "Peak_OffPeak_Ratio",
    "Consumption_Variability",
    "Reading_Completeness_Ratio",
    "Voltage",
    "Current",
    "Power_Factor",
    "Meter_Age",
    "Number_of_Meter_Readings",
]


if __name__ == "__main__":
    import os
    from data_preprocessing import clean_dataset

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity_data.csv")
    df = clean_dataset(data_path)
    df = engineer_features(df)
    print(df[ENGINEERED_FEATURES].describe().T)
