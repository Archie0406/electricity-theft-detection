"""
baseline_model.py
--------------------
A simple, NON-machine-learning baseline used only to answer the
natural interview question: "Why use Machine Learning instead of a
simple rule?"

The rule reuses the same behavioral signal a human investigator might
use as a quick first pass - no learned parameters, no scaling, no
training data at all. It flags a customer-month as anomalous if EITHER:

    - consumption dropped 30% or more vs. the customer's own 3-month
      historical average, OR
    - the absolute deviation from the 6-month historical average
      exceeds 50% of that average.

This mirrors the reasoning already used in assign_risk_level() /
build_reasons() (train_model.py), but collapsed into a single binary
flag with fixed thresholds and no anomaly score, so its performance
can be fairly compared against Isolation Forest, One-Class SVM, and
Random Forest on the exact same test months.

Because it has no "training" step, the baseline is computed directly
on any DataFrame that already has the engineered features - there is
nothing to fit, and therefore nothing that can leak.
"""

import pandas as pd


def baseline_rule_predict(df: pd.DataFrame) -> pd.Series:
    """
    Return a 0/1 Series (1 = flagged as anomalous) using only simple,
    fixed thresholds on features that are already leak-safe (they use
    only past-month information - see feature_engineering.py).
    """
    large_drop = df["Consumption_Change_Pct"] <= -30
    # Average_Consumption_6_Months can be NaN for a customer's first
    # month (no history yet) - treat that as "no baseline to compare
    # against", i.e. never flagged by this rule.
    historical_avg = df["Average_Consumption_6_Months"].fillna(0)
    large_deviation = df["Consumption_Deviation"] > 0.5 * historical_avg

    flagged = (large_drop | large_deviation).astype(int)
    return flagged
