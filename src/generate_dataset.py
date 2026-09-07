"""
generate_dataset.py
--------------------
Generates a SYNTHETIC electricity consumption dataset for the
Electricity Theft Detection project.

Why synthetic data?
Real smart-meter / utility-company consumption data is not publicly
available (utilities keep it private for security and privacy reasons).
For a college project we simulate realistic consumption behavior using
domain knowledge (normal usage patterns, seasonal effects, customer
types) and inject a small, realistic proportion of SIX distinct
suspicious patterns that mimic real-world theft / tampering symptoms:

  1. Sudden drop      - consumption falls sharply in a single month
  2. Gradual drop      - consumption declines slowly over several months
  3. Low power factor   - abnormally low power factor (a real tampering symptom)
  4. Erratic usage      - unusually high month-to-month variability
  5. Peak/off-peak tamper - consumption total looks close to normal, but
                            the split between peak and off-peak hours is
                            heavily skewed (tampering that only affects
                            certain hours of the day)
  6. Combination        - several MILD abnormal indicators together
                            (moderate drop + mildly low power factor +
                            mildly skewed peak ratio), none individually
                            extreme enough to trip a simple single-feature
                            rule - this is the case a multivariate model
                            like Isolation Forest is meant to catch that a
                            simple threshold rule would likely miss.

We also inject a small fraction of otherwise-NORMAL customers with one
month of LEGITIMATE unusual behavior (e.g. vacation, temporary vacancy)
that looks similar to a mild theft pattern but keeps
Theft_Flag_GroundTruth = 0. This is intentional, not noise: real
anomaly detection has to cope with legitimate look-alikes, and it gives
the model (and the student, in an interview) genuine, non-obvious
false-positive material to discuss rather than a suspiciously clean
separation between classes.

The dataset is intentionally NOT perfectly separable - suspicious
records overlap with normal ones, just like in the real world, so
that the ML models have a genuinely non-trivial job.

Note on Average_Consumption_3_Months / Average_Consumption_6_Months:
These are HISTORICAL averages - each row's value is computed only from
that customer's PAST months (via shift(1) before rolling/expanding), so
the current month's own consumption is never included in its own
"historical average". A customer's first month has no prior history and
is left as NaN rather than filled with a made-up number.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_CUSTOMERS = 1500          # unique customers
MONTHS_PER_CUSTOMER = 6     # 6 months of history per customer -> ~9000 rows
SUSPICIOUS_FRACTION = 0.08  # ~8% of customers show suspicious behavior somewhere in their history

LOCATION_TYPES = ["Urban", "Semi-Urban", "Rural"]
CUSTOMER_TYPES = ["Residential", "Commercial", "Industrial"]

# Base consumption ranges (kWh/month) depend on customer type
BASE_CONSUMPTION = {
    "Residential": (120, 350),
    "Commercial": (400, 1200),
    "Industrial": (1500, 5000),
}


def month_seasonal_factor(month):
    """Simple seasonal multiplier (higher usage in summer months for AC load)."""
    summer_months = [4, 5, 6]  # Apr, May, Jun (Indian summer, higher AC usage)
    winter_months = [12, 1, 2]
    if month in summer_months:
        return np.random.normal(1.15, 0.05)
    elif month in winter_months:
        return np.random.normal(0.90, 0.05)
    return np.random.normal(1.0, 0.05)


def generate_customer_profile(customer_id):
    """Create a base profile (fixed characteristics) for one customer."""
    customer_type = np.random.choice(CUSTOMER_TYPES, p=[0.70, 0.22, 0.08])
    location_type = np.random.choice(LOCATION_TYPES, p=[0.5, 0.3, 0.2])
    low, high = BASE_CONSUMPTION[customer_type]
    base_consumption = np.random.uniform(low, high)
    meter_age = np.random.randint(0, 20)  # years
    is_suspicious_customer = np.random.rand() < SUSPICIOUS_FRACTION

    # A separate, smaller group of otherwise-NORMAL customers gets one month
    # of legitimately unusual behavior (e.g. vacation, temporary closure) -
    # ground truth stays 0 (it is NOT theft), but it looks similar to a mild
    # "gradual_drop" theft pattern. This is deliberate: real normal customers
    # sometimes have one odd month for innocent reasons, and a model that
    # only ever sees clean "normal" data would look artificially good. This
    # is also what creates genuine false positives to discuss in Section 13
    # (Error Analysis) instead of the dataset being trivially separable.
    has_legit_anomaly = (not is_suspicious_customer) and (np.random.rand() < 0.06)

    return {
        "Customer_ID": f"C{customer_id:05d}",
        "Customer_Type": customer_type,
        "Location_Type": location_type,
        "base_consumption": base_consumption,
        "Meter_Age": meter_age,
        "is_suspicious_customer": is_suspicious_customer,
        "has_legit_anomaly": has_legit_anomaly,
        "legit_anomaly_month": np.random.randint(2, MONTHS_PER_CUSTOMER) if has_legit_anomaly else None,
    }


def generate_month_record(profile, month_index, history):
    """Generate one month's record for a customer, given prior history."""
    month = (month_index % 12) + 1
    seasonal = max(month_seasonal_factor(month), 0.5)

    base = profile["base_consumption"]

    # Natural month-to-month noise for a normal customer
    noise = np.random.normal(1.0, 0.08)
    consumption = base * seasonal * noise

    theft_flag = 0  # ground-truth label used ONLY for evaluation, not shown to the model as a feature
    reason = "normal"

    # Inject suspicious behavior for a subset of months belonging to "suspicious" customers.
    # Not every month is suspicious - theft/tampering usually starts at some point.
    # Six distinct pattern types are used so the problem isn't reducible to one
    # simple rule (e.g. "flag every big drop") - a real investigator would also
    # see this variety, and it's what makes the ML models' job non-trivial.
    if profile["is_suspicious_customer"] and month_index >= 2 and np.random.rand() < 0.55:
        theft_flag = 1
        pattern = np.random.choice(
            ["sudden_drop", "gradual_drop", "low_pf_tamper", "erratic",
             "peak_offpeak_shift", "combination"],
            p=[0.25, 0.20, 0.18, 0.12, 0.13, 0.12],
        )
        if pattern == "sudden_drop":
            reason = "sudden_drop"
            consumption = consumption * np.random.uniform(0.15, 0.45)
        elif pattern == "gradual_drop":
            reason = "gradual_drop"
            # gradual decline relative to their own recent history
            if history:
                consumption = history[-1]["Electricity_Consumption_kWh"] * np.random.uniform(0.75, 0.90)
            else:
                consumption = consumption * np.random.uniform(0.75, 0.90)
        elif pattern == "low_pf_tamper":
            reason = "low_pf_tamper"
            consumption = consumption * np.random.uniform(0.5, 0.8)
        elif pattern == "erratic":
            reason = "erratic"
            consumption = consumption * np.random.uniform(0.3, 1.6)
        elif pattern == "peak_offpeak_shift":
            # Total consumption looks close to normal, but the split between
            # peak and off-peak hours is distorted - a real symptom of meter
            # tampering that only affects certain hours (e.g. bypassing the
            # meter only during peak-tariff hours). Deliberately mild on the
            # total-consumption feature so it isn't obvious from that alone.
            reason = "peak_offpeak_shift"
            consumption = consumption * np.random.uniform(0.85, 1.05)
        elif pattern == "combination":
            # A milder version of several indicators together, rather than
            # one extreme signal - more realistic than an obvious single
            # giveaway, and harder for a simple one-rule baseline to catch.
            reason = "combination"
            consumption = consumption * np.random.uniform(0.55, 0.75)
    elif profile.get("has_legit_anomaly") and (month_index + 1) == profile.get("legit_anomaly_month"):
        # A LEGITIMATE but unusual month for an otherwise-normal customer
        # (e.g. vacation, temporary closure, seasonal absence). Ground
        # truth stays 0 - this is explicitly NOT theft - but the drop is
        # similar in size to a mild "gradual_drop" theft pattern, so it
        # creates genuine overlap between the two classes instead of the
        # dataset being trivially separable. This is also what produces
        # real false positives to discuss in the Error Analysis section,
        # rather than the classes being cleanly separated by construction.
        reason = "legit_anomaly"
        consumption = consumption * np.random.uniform(0.30, 0.60)

    consumption = max(consumption, 5)  # consumption can't be negative / near-zero realistically

    # Voltage / current / power factor - normally stable, but tampering can distort power factor
    voltage = np.random.normal(230, 5)  # typical Indian household voltage ~230V
    if reason in ("low_pf_tamper", "combination"):
        power_factor = np.clip(np.random.normal(0.55, 0.08), 0.3, 0.75)
    else:
        power_factor = np.clip(np.random.normal(0.92, 0.04), 0.6, 0.99)

    # current derived approximately from P = V * I * PF (kept simple, in Amps, monthly avg)
    # Using a simplified relation scaled down for realism at household meter level
    current = (consumption * 1000) / (voltage * power_factor * 24 * 30) if consumption > 0 else 0
    current = max(current, 0.1)

    if reason == "peak_offpeak_shift":
        # Push a larger-than-normal share into off-peak hours (mimics
        # under-reporting during metered peak-tariff hours).
        peak_ratio = np.random.uniform(0.25, 0.40)
    elif reason == "combination":
        peak_ratio = np.random.uniform(0.30, 0.45)
    else:
        peak_ratio = np.random.uniform(0.55, 0.75)
    peak_consumption = consumption * peak_ratio
    off_peak_consumption = consumption - peak_consumption

    # Number of meter readings recorded this month (missed readings can indicate tampering/access issues)
    if reason in ("sudden_drop", "erratic", "combination"):
        num_readings = np.random.choice([15, 20, 25, 28, 30], p=[0.15, 0.2, 0.25, 0.2, 0.2])
    else:
        num_readings = np.random.choice([28, 29, 30], p=[0.2, 0.3, 0.5])

    return {
        "Consumption": consumption,
        "Voltage": voltage,
        "Current": current,
        "Power_Factor": power_factor,
        "Peak_Consumption": peak_consumption,
        "Off_Peak_Consumption": off_peak_consumption,
        "Number_of_Meter_Readings": num_readings,
        "theft_flag": theft_flag,
        "reason": reason,
    }


def introduce_missing_and_dirty_data(df):
    """Introduce a small amount of realistic messiness: missing values, duplicates."""
    df = df.copy()

    # ~1.5% missing values in a few columns (sensor/meter reading gaps)
    for col in ["Voltage", "Power_Factor", "Current"]:
        mask = np.random.rand(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # A handful of duplicate rows (data-entry duplication, common in real utility exports)
    dup_rows = df.sample(frac=0.005, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


def build_dataset():
    records = []
    for cust_idx in range(1, N_CUSTOMERS + 1):
        profile = generate_customer_profile(cust_idx)
        history = []
        prev_consumption = None

        for m in range(MONTHS_PER_CUSTOMER):
            month_data = generate_month_record(profile, m, history)

            row = {
                "Customer_ID": profile["Customer_ID"],
                "Month": m + 1,
                "Electricity_Consumption_kWh": round(month_data["Consumption"], 2),
                "Previous_Month_Consumption": round(prev_consumption, 2) if prev_consumption is not None else np.nan,
                "Peak_Consumption": round(month_data["Peak_Consumption"], 2),
                "Off_Peak_Consumption": round(month_data["Off_Peak_Consumption"], 2),
                "Voltage": round(month_data["Voltage"], 2),
                "Current": round(month_data["Current"], 3),
                "Power_Factor": round(month_data["Power_Factor"], 3),
                "Meter_Age": profile["Meter_Age"],
                "Number_of_Meter_Readings": month_data["Number_of_Meter_Readings"],
                "Location_Type": profile["Location_Type"],
                "Customer_Type": profile["Customer_Type"],
                # ground truth - kept ONLY for evaluation purposes (simulates investigator feedback)
                "Theft_Flag_GroundTruth": month_data["theft_flag"],
            }
            history.append({"Electricity_Consumption_kWh": row["Electricity_Consumption_kWh"]})
            prev_consumption = row["Electricity_Consumption_kWh"]
            records.append(row)

    df = pd.DataFrame(records)

    # Compute HISTORICAL averages using only PAST months, never the current
    # month's own consumption. This is critical for anomaly detection: if a
    # customer's current month were included in their own "historical
    # average", a sudden theft-like drop would drag the average down with
    # it, hiding the very deviation we're trying to detect (and would count
    # as data leakage - using a value's own future/current information to
    # describe its own history).
    #
    #   shift(1)  ->  moves each customer's series down by one row, so
    #                 whatever lines up with "Month N" is actually the
    #                 value from "Month N-1" - i.e. everything BEFORE the
    #                 current row.
    #   .rolling()/.expanding() then only ever see that shifted, past-only
    #                 series.
    #
    # Example: Jan=280, Feb=290, Mar=100
    #   March's 3-month average = mean(Jan, Feb) = 285   <- NOT (280+290+100)/3
    #
    # A customer's very FIRST month has no prior history at all, so both
    # columns are legitimately NaN there (not an artificial 0 or a value
    # that includes itself) - this is handled downstream in
    # feature_engineering.py, which converts "no history yet" into an
    # explicit, documented default rather than silently faking a number.
    df = df.sort_values(["Customer_ID", "Month"]).reset_index(drop=True)
    df["Average_Consumption_3_Months"] = (
        df.groupby("Customer_ID")["Electricity_Consumption_kWh"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["Average_Consumption_6_Months"] = (
        df.groupby("Customer_ID")["Electricity_Consumption_kWh"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    df = introduce_missing_and_dirty_data(df)

    # Shuffle rows (real exports are not customer-sorted) but keep reproducibility
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset = build_dataset()
    import os
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "electricity_data.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Dataset generated: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(f"Suspicious (ground truth) proportion: {dataset['Theft_Flag_GroundTruth'].mean():.3f}")
    print(f"Saved to {output_path}")
