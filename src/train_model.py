"""
train_model.py
----------------
Trains the anomaly-detection models used for electricity theft
detection, using a TIME-AWARE pipeline that avoids data leakage.

Why anomaly detection instead of plain classification?
---------------------------------------------------------
In the real world, confirmed electricity-theft labels are RARE and
expensive to obtain (they usually only exist after a physical
inspection). Most customers are never inspected, so we cannot assume
the "normal" majority is truly 100% clean, and we cannot assume we
have enough confirmed theft cases to train a reliable supervised
classifier. Anomaly detection is the standard approach in this
situation: instead of learning "theft vs not theft" from labels, the
model learns what NORMAL consumption behavior looks like and flags
customers whose behavior deviates significantly from that norm.

Models used:
    1. Isolation Forest (PRIMARY model)
       - Works by randomly partitioning the data; anomalies are
         "easier to isolate" (require fewer splits) than normal points.
       - Efficient, works well with mixed-scale numeric features,
         and does not assume any particular data distribution.

    2. One-Class SVM (COMPARISON model)
       - Learns a boundary around the "normal" region in feature space.
       - Sensitive to feature scaling, so it needs StandardScaler.
       - Included to show the student can compare multiple approaches.

    3. Random Forest Classifier (OPTIONAL, supervised comparison)
       - Only trained because our synthetic dataset happens to include
         a ground-truth label (simulating past investigation outcomes).
       - In a real deployment this would usually NOT be available for
         most customers, so this is shown purely as a comparison to
         illustrate the difference between supervised and unsupervised
         approaches - not as the main solution.

A SIMPLE NON-ML BASELINE (see baseline_model.py) is also computed for
every row, using fixed threshold rules on the same engineered features
(no training, no learned parameters at all). This exists to answer a
natural interview question - "why use ML instead of a simple rule?" -
with real numbers rather than an assumption. See evaluate_model.py for
the actual baseline-vs-ML comparison.

IMPORTANT - ML output vs risk classification are two separate steps:

    ML Model  ->  Anomaly Score  ->  Explainable Risk Rules  ->
    Normal / Suspicious / High Risk

    Isolation Forest only produces a raw anomaly score. The
    Normal/Suspicious/High Risk labels come from simple, PROJECT-DEFINED
    threshold rules applied to that score (see assign_risk_level below) -
    they are not something the model "decides" on its own. This
    distinction matters: the system identifies anomalous consumption
    patterns that MAY indicate potential electricity theft and
    prioritizes them for further human investigation - it does not,
    and cannot, prove theft on its own.

TIME-AWARE TRAIN/TEST SPLIT - avoiding data leakage:
    The dataset contains 6 months of history per customer. Splitting
    rows randomly would let a customer's LATER month end up in the
    training set while an EARLIER month for the SAME customer ends up
    in the test set - the model would effectively be tested on data
    from "the past" relative to what it trained on, which never
    happens in a real deployment. Instead, this pipeline uses:

        Months 1-4  ->  Training (the model's "historical" data)
        Months 5-6  ->  Testing  (simulates new/unseen consumption)

    All preprocessing statistics (missing-value medians) and the
    feature scaler are FIT ONLY on the training months, then reused
    (never re-fit) on the test months - exactly like calling
    .fit() then .transform() with a scikit-learn Pipeline.

ANOMALY SCORE NORMALIZATION - also leak-safe:
    Isolation Forest's raw score_samples() output is converted into an
    intuitive 0-1 "anomaly score" via min-max normalization. The min/max
    values used for this are computed ONLY from the TRAINING months'
    raw scores (fit_score_normalizer), then reused to normalize every
    row - train, test, and (in principle) any new customer data the
    dashboard would show (apply_score_normalizer). The min/max are
    saved to models/anomaly_score_params.pkl so the same normalization
    is reproducible outside this script. Computing min/max from the
    full train+test dataset would let test-month scores influence the
    scale applied to training-month scores - a subtle leak that doesn't
    affect the model itself, but affects the honesty of the numbers
    the model reports.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM

from data_preprocessing import (
    load_data,
    remove_duplicates,
    fix_data_types,
    time_based_split,
    fit_missing_value_stats,
    apply_missing_value_stats,
    TRAIN_MONTHS,
    TEST_MONTHS,
)
from feature_engineering import engineer_features, ENGINEERED_FEATURES, compute_peak_offpeak_fallback
from baseline_model import baseline_rule_predict

RANDOM_SEED = 42

# Contamination = expected proportion of anomalies in the data.
# We set this based on domain assumption / typical utility theft-rate
# estimates (a few percent), not on the ground-truth label, since in a
# real deployment we would NOT have that label available.
CONTAMINATION = 0.05

# Safety check: the ground-truth label must NEVER be used as an input
# feature for the anomaly-detection models - it is for evaluation only.
assert "Theft_Flag_GroundTruth" not in ENGINEERED_FEATURES, (
    "Data leakage: ground-truth label must not appear in the feature list!"
)


def train_isolation_forest(X_train_scaled):
    """Fit Isolation Forest on TRAINING months only (months 1-4)."""
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)
    return model


def train_one_class_svm(X_train_scaled):
    """Fit One-Class SVM on TRAINING months only (months 1-4)."""
    model = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
    model.fit(X_train_scaled)
    return model


def train_random_forest(X_train, y_train):
    """
    Optional supervised comparison model. Trained ONLY on the training
    months (1-4); it is evaluated separately on the held-out test
    months (5-6) in evaluate_model.py - a genuine time-based
    train/test split, not a random one.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",  # important because theft cases are rare (class imbalance)
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_raw_anomaly_scores(iso_model, X_scaled):
    """
    Return Isolation Forest's raw anomaly scores, flipped so that HIGHER
    means MORE anomalous (score_samples() itself returns higher values
    for NORMAL points, which is the opposite of what we want to display).
    These raw scores are NOT yet on a 0-1 scale.
    """
    raw_scores = iso_model.score_samples(X_scaled)  # higher = more normal
    return -raw_scores  # higher = more anomalous


def fit_score_normalizer(raw_train_scores):
    """
    Compute min-max normalization parameters using ONLY the raw anomaly
    scores from the TRAINING months. This is the leak-safe equivalent of
    fitting a scaler: the min/max are learned once, from training data,
    and then reused everywhere else (test months, the full scored
    dataset, and any future/new customer scored by the dashboard).

    Using the full dataset's min/max (train + test combined) would let
    the test set influence how training-time scores get displayed, and
    would mean the normalization itself is not something a real
    deployment could reproduce for genuinely new, unseen data.
    """
    score_min = float(np.min(raw_train_scores))
    score_max = float(np.max(raw_train_scores))
    return {"score_min": score_min, "score_max": score_max}


def apply_score_normalizer(raw_scores, score_params):
    """
    Normalize raw anomaly scores to a 0-1 scale using PRE-COMPUTED
    TRAINING min/max (never recomputed from the data being scored).

    Test-time (or dashboard) raw scores can fall slightly outside the
    training min/max range - that's expected, since test data is
    genuinely new. We clip the normalized result to [0, 1] so risk
    thresholds (see assign_risk_level) stay meaningful instead of
    producing scores below 0 or above 1.
    """
    score_min = score_params["score_min"]
    score_max = score_params["score_max"]
    denom = score_max - score_min

    raw_scores = np.asarray(raw_scores, dtype=float)

    if denom == 0:
        # Degenerate case: every training score was identical (would only
        # happen with a tiny or artificial dataset). Avoid dividing by
        # zero - return a neutral mid-scale score for everything since
        # there's no real spread to normalize against.
        return np.full_like(raw_scores, 0.5)

    normalized = (raw_scores - score_min) / denom
    return np.clip(normalized, 0.0, 1.0)


def assign_risk_level(row):
    """
    Convert an Isolation Forest anomaly score into a simple, explainable
    3-tier risk category using PROJECT-DEFINED RULES (not something the
    model learns on its own):

        Isolation Forest -> Anomaly Score -> these threshold rules ->
        Normal / Suspicious / High Risk

    Thresholds are intentionally simple and documented (not a black
    box) so the student can explain and justify them:

    - anomaly_score >= 0.75  AND large consumption drop  -> High Risk
    - anomaly_score >= 0.55                               -> Suspicious
    - otherwise                                           -> Normal

    Using both the anomaly score AND a behavioral indicator (consumption
    change) avoids flagging "High Risk" purely because of one noisy
    score, and mirrors how a real investigator would reason. These
    exact threshold values (0.75 / 0.55 / -30%) are reasonable starting
    points chosen for explainability, not values learned from data -
    in a real deployment they would be tuned against feedback from
    actual field investigations over time.
    """
    score = row["Anomaly_Score"]
    change_pct = row["Consumption_Change_Pct"]

    if score >= 0.75 and change_pct <= -30:
        return "High Risk"
    elif score >= 0.55:
        return "Suspicious"
    else:
        return "Normal"


def build_reasons(row):
    """Generate simple, human-readable reasons for why a record was flagged."""
    reasons = []
    if row["Consumption_Change_Pct"] <= -30:
        reasons.append("Large drop from recent average consumption")
    if row["MoM_Change_Pct"] <= -40:
        reasons.append("Sharp single-month consumption drop vs. previous month")
    if row["Consumption_Deviation"] > row["Average_Consumption_6_Months"] * 0.5:
        reasons.append("Large deviation from 6-month historical average")
    if row["Power_Factor"] < 0.7:
        reasons.append("Abnormally low power factor (possible tampering indicator)")
    if row["Consumption_Variability"] > 0.3:
        reasons.append("High month-to-month consumption variability")
    if row["Reading_Completeness_Ratio"] < 0.8:
        reasons.append("Incomplete / missing meter readings this month")
    if not reasons:
        reasons.append("Minor deviation from typical consumption pattern")
    return "; ".join(reasons)


def run_training_pipeline(data_path, models_dir):
    print("Loading raw data...")
    raw_df = load_data(data_path)
    raw_df = remove_duplicates(raw_df)
    raw_df = fix_data_types(raw_df)

    print(f"\nCreating time-based split (train months={TRAIN_MONTHS}, test months={TEST_MONTHS})...")
    train_raw, test_raw = time_based_split(raw_df)

    print("\nFitting missing-value statistics on TRAIN months only, applying to both splits...")
    impute_stats = fit_missing_value_stats(train_raw)
    train_raw = apply_missing_value_stats(train_raw, impute_stats)
    test_raw = apply_missing_value_stats(test_raw, impute_stats)

    # Peak_OffPeak_Ratio's fallback (for undefined ratios caused by zero
    # off-peak consumption) must ALSO be computed from TRAIN rows only -
    # otherwise the fallback value would be influenced by test-month
    # data, the same category of leak that StandardScaler/imputation
    # already avoid. Computed here (on raw train rows, before the ratio
    # itself exists) and passed into engineer_features() below.
    peak_offpeak_fallback = compute_peak_offpeak_fallback(train_raw)
    impute_stats["Peak_OffPeak_Ratio_fallback"] = peak_offpeak_fallback
    print(f"Peak/Off-Peak ratio fallback (median, TRAIN months only): {peak_offpeak_fallback:.4f}")

    # Recombine into one time-ordered frame so that per-customer history
    # based features (rolling averages, historical variability) can see
    # each customer's full timeline. This is NOT leakage: for a test-month
    # row, the "history" it looks back on is genuinely earlier months
    # (train months) - exactly the information available in production.
    full_df = pd.concat([train_raw, test_raw]).sort_values(["Customer_ID", "Month"]).reset_index(drop=True)

    print("\nEngineering features (each row only uses its own past/current-month data)...")
    full_df = engineer_features(full_df, peak_offpeak_fallback=peak_offpeak_fallback)

    full_df["Data_Split"] = np.where(full_df["Month"].isin(TRAIN_MONTHS), "Train", "Test")
    train_mask = full_df["Data_Split"] == "Train"
    test_mask = full_df["Data_Split"] == "Test"

    X_all = full_df[ENGINEERED_FEATURES]
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train = full_df.loc[train_mask, "Theft_Flag_GroundTruth"]
    y_test = full_df.loc[test_mask, "Theft_Flag_GroundTruth"]

    print(f"\nFeature matrix: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {X_train.shape[1]} features")

    print("Fitting StandardScaler on TRAIN features only...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X_all)  # for scoring every row for the dashboard

    print("\nTraining Isolation Forest (primary model) on historical (train) months...")
    iso_model = train_isolation_forest(X_train_scaled)

    print("Training One-Class SVM (comparison model) on historical (train) months...")
    ocsvm_model = train_one_class_svm(X_train_scaled)

    print("Training Random Forest (optional supervised comparison) on historical (train) months...")
    rf_model = train_random_forest(X_train, y_train)

    # ---- Anomaly score normalization (leak-safe) ----
    # 1. Get raw scores for every row (train + test) from the trained model.
    # 2. Fit the min/max normalization parameters using ONLY the rows that
    #    belong to the TRAINING months - never the test months, and never
    #    the combined train+test set.
    # 3. Apply that SAME train-derived min/max to every row (train, test,
    #    and later, any new data the dashboard would show), exactly like
    #    reusing a fitted StandardScaler.
    raw_scores_all = compute_raw_anomaly_scores(iso_model, X_all_scaled)
    raw_scores_train = raw_scores_all[train_mask.values]

    score_params = fit_score_normalizer(raw_scores_train)
    print(
        f"\nAnomaly score normalization fit on TRAIN months only: "
        f"min={score_params['score_min']:.4f}, max={score_params['score_max']:.4f}"
    )

    full_df["Anomaly_Score"] = apply_score_normalizer(raw_scores_all, score_params)
    full_df["IsolationForest_Prediction"] = iso_model.predict(X_all_scaled)  # -1 = anomaly, 1 = normal
    full_df["OneClassSVM_Prediction"] = ocsvm_model.predict(X_all_scaled)
    full_df["RandomForest_Prediction"] = rf_model.predict(X_all)

    # Simple non-ML baseline (see baseline_model.py) - computed the same
    # way for every row, using only already leak-safe engineered features.
    # Used purely as a reference point to justify why ML is worth using.
    full_df["Baseline_Prediction"] = baseline_rule_predict(full_df)

    # Risk classification (rule-based conversion of the anomaly score - see assign_risk_level docstring)
    full_df["Risk_Level"] = full_df.apply(assign_risk_level, axis=1)
    full_df["Flag_Reasons"] = full_df.apply(build_reasons, axis=1)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(iso_model, os.path.join(models_dir, "isolation_forest.pkl"))
    joblib.dump(ocsvm_model, os.path.join(models_dir, "one_class_svm.pkl"))
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(ENGINEERED_FEATURES, os.path.join(models_dir, "feature_list.pkl"))
    joblib.dump(impute_stats, os.path.join(models_dir, "impute_stats.pkl"))
    joblib.dump(score_params, os.path.join(models_dir, "anomaly_score_params.pkl"))

    # Save the scored dataset (with a Data_Split column) for the
    # dashboard and evaluation script to use directly.
    scored_path = os.path.join(os.path.dirname(data_path), "scored_customers.csv")
    full_df.to_csv(scored_path, index=False)

    print("\nTraining complete.")
    print(f"Models saved to: {models_dir}")
    print(f"Scored dataset saved to: {scored_path}")
    print("\nRisk level distribution (all customer-months):")
    print(full_df["Risk_Level"].value_counts())

    rf_test_data = (X_test, y_test)
    return full_df, iso_model, ocsvm_model, rf_model, rf_test_data, scaler, score_params


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "electricity_data.csv")
    models_dir = os.path.join(base_dir, "..", "models")

    run_training_pipeline(data_path, models_dir)
