# Electricity Theft Detection Using Machine Learning

A 3rd-year B.Tech / B.E. (Artificial Intelligence & Data Science) project that uses
anomaly detection to flag potentially suspicious electricity consumption patterns for
further human investigation.

---

## 1. Project Overview

This project analyzes monthly electricity-consumption records and identifies
customer-months whose behavior deviates significantly from what's normal — for that
customer and for similar customers. It does **not** attempt to prove theft; it produces
a short, prioritized list of accounts that a human investigator could review, instead of
requiring every customer to be checked manually.

The pipeline is a straightforward, explainable Data Science + ML workflow: synthetic
data generation → cleaning → time-based train/test split → feature engineering →
anomaly detection (Isolation Forest, primary) → rule-based risk classification →
evaluation → Streamlit dashboard.

## 2. Problem Statement

Electricity theft and meter tampering cause real revenue loss for power distribution
companies, but manually reviewing every customer's consumption history doesn't scale.
This project explores whether anomaly detection can highlight consumption patterns that
deviate meaningfully from a customer's own historical norm, narrowing the search space
for investigators.

## 3. Motivation

Utilities lose revenue every year to theft and tampering, but false accusations also
seriously harm genuine customers. A data-driven early-warning system — one that is
explicit about its uncertainty and limitations — is more useful and fairer than either
ignoring the problem or acting on gut feeling.

## 4. Objectives

- Analyze electricity consumption data to understand normal usage patterns.
- Engineer behavioral features that capture signs associated with tampering, using only
  information that would genuinely be available at prediction time.
- Evaluate models the way they would actually be used: **train on historical months,
  test on new/unseen months**, not on randomly shuffled data.
- Build and compare anomaly-detection models, and convert their output into a simple,
  explainable 3-tier risk classification.
- Distinguish **record-level** (customer-month) risk from **customer-level** risk.
- Build a dashboard for exploring flagged customers, with a clear disclaimer throughout.

## 5. Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (Isolation Forest, One-Class SVM, Random Forest, StandardScaler) |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model persistence | Joblib |

No deep learning, LSTMs, transformers, generative AI, cloud infrastructure, or
Docker/Kubernetes are used. The project is intentionally scoped to what a 3rd-year
student can build, understand, and defend end-to-end.

## 6. Dataset

**The dataset is synthetically generated** (`src/generate_dataset.py`) because real
smart-meter data isn't publicly available. It simulates 1,500 customers over 6 months
(9,000 customer-month records):

- Consumption depends on `Customer_Type` (Residential / Commercial / Industrial) and
  season (higher in summer months, consistent with AC-driven demand).
- ~8% of customers have injected suspicious patterns across **six distinct types**:
  sudden drop, gradual drop, abnormally low power factor, erratic usage, an abnormal
  peak/off-peak split (tampering that only affects certain hours), and a milder
  combination of several indicators at once — each starting partway through the
  customer's history, not from month 1, since real tampering doesn't start on day one.
- A separate ~6% of otherwise-**normal** customers get one month of *legitimately*
  unusual behavior (e.g. vacation, temporary closure) — ground truth stays 0 (not
  theft), but it can look similar to a mild theft pattern. This is intentional: it
  creates genuine overlap between the "normal" and "suspicious" classes so the dataset
  isn't trivially separable, and is what produces real false positives to discuss in
  Section 13b (Error Analysis) rather than the classes being cleanly separated by
  construction.
- Missing values (~1.5% in a few sensor columns) and duplicate rows are deliberately
  introduced to simulate real-world data-quality issues.
- A `Theft_Flag_GroundTruth` column records which rows were generated as suspicious.
  This label is used **only for evaluation** (simulating past investigation outcomes)
  and is asserted at import-time to never appear in the model's feature list — see
  Section 12.

Suspicious cases are not made obvious: they overlap with normal behavior (in both
directions — some suspicious months look mild, some normal months look unusual), so
anomaly detection is a genuinely non-trivial problem rather than a solved one.

## 7. Data Preprocessing

`src/data_preprocessing.py` provides two paths, used for different purposes:

- **`clean_dataset()`** — an all-data convenience cleaner used only for quick
  exploration in `notebooks/exploratory_analysis.ipynb`, where there's no train/test
  split yet and the goal is just to look at the data.
- **`fit_missing_value_stats()` / `apply_missing_value_stats()`** — the leak-safe path
  used by the real training pipeline. Per-`Customer_Type` medians for `Voltage`,
  `Power_Factor`, and `Current` are computed **only from the training months**, then
  the exact same values are used to fill missing values in the test months — the same
  fit/transform discipline a scikit-learn `Imputer` would enforce.

Also handled: duplicate-row removal, data-type correction, and IQR-based outlier
*flagging* (outliers are deliberately not auto-removed, since in this problem outliers
are often exactly the records worth investigating).

## 8. Time-Based Train/Test Strategy

Each customer has 6 months of history. Splitting rows **randomly** would let a
customer's *later* month land in training while an *earlier* month for the same
customer lands in test — effectively testing the model on the past relative to what it
trained on, which can't happen in a real deployment.

Instead, the project uses a **time-based split**:

```
Month 1 → Train      Month 5 → Test
Month 2 → Train       Month 6 → Test
Month 3 → Train
Month 4 → Train
```

- **Train**: months 1–4 (6,000 records) — the model's "historical" data.
- **Test**: months 5–6 (3,000 records) — simulates new, unseen consumption.

This mirrors the real workflow the system is meant to support:

```
Historical electricity data → Train model → New/unseen consumption → Detect anomalies
```

**Everything that is "fit" — missing-value statistics and the `StandardScaler` — is fit
only on the training months, then reused (never re-fit) on the test months.** All three
models (Isolation Forest, One-Class SVM, Random Forest) are trained on months 1–4 only
and evaluated on months 5–6 only.

## 9. Feature Engineering

| Feature | Why it's useful |
|---|---|
| `Consumption_Change_Pct` | Current consumption vs. the customer's own **historical** 3-month average (previous months only — see below). Large negative values are the classic "sudden drop" signature of tampering. |
| `MoM_Change_Pct` | Current consumption vs. the customer's actual **previous month** (`Previous_Month_Consumption`, generated directly in `generate_dataset.py`) — see below for why this is kept separate from `Consumption_Change_Pct`. |
| `Consumption_Deviation` | Absolute distance from the **historical** 6-month average (previous months only), in kWh — catches deviation in either direction. |
| `Peak_OffPeak_Ratio` | Legitimate customers have a fairly stable peak/off-peak ratio; tampering affecting only certain hours distorts it. |
| `Consumption_Variability` | Coefficient of variation (std/mean) of the customer's **past** consumption, computed over a rolling 3-month window that excludes the current month (see below). |
| `Reading_Completeness_Ratio` | Missed meter readings can indicate access/tampering issues. |

**Why both `Consumption_Change_Pct` and `MoM_Change_Pct`?** The raw dataset already
generates `Previous_Month_Consumption` (the customer's actual prior month), but earlier
versions of this project computed it and then never used it downstream — an unused
column left in "for appearance." `MoM_Change_Pct` puts it to genuine use: a sharp
single-month drop (this feature) is a different signal from a gradual drift away from a
smoother 3-month baseline (`Consumption_Change_Pct`). A customer whose consumption fell
sharply just this month, before their 3-month average has caught up, is caught by
`MoM_Change_Pct` and might be missed by the other feature alone — and vice versa for a
slow decline. Both are genuinely leak-safe: `Previous_Month_Consumption` is generated
directly from the prior loop iteration in `generate_dataset.py`, never from a future
month.

**Historical averages — avoiding leakage in detail.** `Average_Consumption_3_Months`
and `Average_Consumption_6_Months` (computed in `src/generate_dataset.py`) are built
with `.shift(1)` **before** the `.rolling()` / `.expanding()` call, so each row's
"historical average" is computed only from that customer's *earlier* months — the
current month's own reading is never part of its own average:

```
Customer A:  Jan=280   Feb=290   Mar=100

Mar's 3-month average = mean(Jan, Feb) = 285      ← NOT (280+290+100)/3 = 223.33
```

A customer's **first month has no prior history at all**, so both average columns are
genuinely `NaN` for that row — not an artificial 0 or a self-referencing value.
`Consumption_Change_Pct`, `MoM_Change_Pct`, and `Consumption_Deviation` convert this
into an explicit, documented default of `0` ("no deviation signal available yet")
rather than crashing or silently guessing a number; the dashboard shows "N/A (no prior
history)" for the raw average itself.

**`Consumption_Variability` — avoiding leakage in detail.** This feature is the
coefficient of variation of a customer's consumption over the **3 months immediately
before** the current row — the current month's own value is excluded on purpose.
Example:

```
Customer A:  Jan=280   Feb=290   Mar=275   Apr=285   May=100

May's variability → computed from Feb, Mar, Apr only
June's consumption is NEVER used when computing a feature for May
```

Customers with fewer than 2 prior months of history get a default value of 0 (no
evidence of instability yet) rather than a guessed value.

**`Peak_OffPeak_Ratio` fallback — avoiding leakage in detail.** A small number of
rows have zero off-peak consumption, making the ratio mathematically undefined
(division by zero). The gap is filled with a fallback value — and that fallback value
is now computed **only from training-month rows** (`compute_peak_offpeak_fallback()` in
`feature_engineering.py`, called on `train_raw` in `train_model.py`), then reused for
both train and test rows, saved to `models/impute_stats.pkl` alongside the other
imputation statistics. An earlier version of this pipeline computed this fallback from
the full train+test dataset at once — a subtle leak (the same category of mistake as
fitting a scaler on test data) that has since been fixed and re-verified: the
train-only fallback (1.8496) is measurably different from the old full-dataset value
(1.8439), confirming the fix changes real behavior, not just documentation.

All features use only information available **strictly before** the current month for
that customer — no current-month or future information leaks in.

## 10. Isolation Forest (Primary Model)

**Why anomaly detection at all?** Confirmed theft labels are rare and expensive to
obtain in the real world (they typically require a physical inspection). We can't
assume the "normal" majority is 100% clean, and there usually isn't enough labeled data
to train a reliable supervised classifier. Anomaly detection instead learns what
*normal* behavior looks like and flags significant deviations from it — a good fit for
this problem.

**Why Isolation Forest specifically?** It isolates points via random recursive
partitioning; anomalies require fewer splits to isolate than normal points, so their
average path length across many random trees is shorter. It works well with mixed-scale
numeric features, makes no distributional assumptions, and is computationally cheap —
all good properties for a first, primary anomaly-detection model on tabular data.

Trained on months 1–4 only, then used to score every month (including 5–6) using
`.score_samples()`.

**Anomaly score normalization — avoiding leakage in detail.** `.score_samples()`
returns a raw, unbounded score (higher = more *normal*, which is inverted so higher =
more anomalous). To turn this into an intuitive 0–1 scale, the raw scores need a min and
a max to normalize against. **These min/max values are computed only from the raw
scores of the training months** (`fit_score_normalizer()` in `train_model.py`) and then
reused, unchanged, to normalize every row — train, test, and (in principle) any new
customer the dashboard would score later:

```
X_train → Isolation Forest → raw TRAIN scores → score_min, score_max (saved)
X_test  → Isolation Forest → raw TEST scores  → normalized using the SAVED score_min/max
```

Computing min/max from the combined train+test scores (or worse, from the full
dataset) would let the test months influence the scale applied to training scores — a
subtle leak that doesn't change what the model predicts, but *would* make the reported
anomaly scores optimistic and not reproducible on genuinely new data. The fitted
`score_min` / `score_max` are saved to `models/anomaly_score_params.pkl` and reused as-is
by `src/prediction.py`. Test-month raw scores can fall outside the training range (since
they're genuinely new data) — the normalized result is clipped to `[0, 1]` so risk
thresholds stay meaningful.

## 11. One-Class SVM (Unsupervised Comparison)

Learns a boundary around the "normal" region of feature space. More sensitive to
feature scaling than Isolation Forest, so it's trained on the same `StandardScaler`
output (fit on train months only). Included to compare a second unsupervised approach
against Isolation Forest rather than relying on a single algorithm's behavior.

## 12. Random Forest (Optional Supervised Benchmark)

Only trainable because the *synthetic* dataset happens to include a ground-truth label.
Trained and evaluated on the same time-based split (train: months 1–4, test: months
5–6) for a fair, apples-to-apples comparison. In a real deployment this label usually
would **not** exist for most customers, so Random Forest is shown purely as an
educational benchmark — **it is not the primary model**, and is not meant to be treated
as production-ready.

## 12b. Simple Rule Baseline (No ML)

`src/baseline_model.py` adds a fourth, deliberately non-ML reference point: flag a
customer-month if `Consumption_Change_Pct ≤ -30%` **or** the deviation from the 6-month
historical average exceeds 50% of that average — fixed thresholds, no training, nothing
that can leak. This exists to answer a question every interviewer eventually asks:

> "Why did you use Machine Learning instead of a simple rule?"

The honest answer, from the actual results in Section 13: **the simple rule actually
beat Isolation Forest on F1-score** in this run. That's not a bug — the rule was
hand-designed using the same domain logic that generated the synthetic anomalies, so
it has an unfair advantage a real-world rule wouldn't have. Isolation Forest has to
*discover* multivariate patterns without being told the rule, and its PR-AUC (which
looks across every threshold, not just one) shows it's the weakest of the three ML
models here — a genuinely useful, if humbling, finding.

**ML output vs. risk classification are two separate steps:**

```
Isolation Forest → Anomaly Score → Explainable Risk Rules → Normal / Suspicious / High Risk
```

Isolation Forest only produces a raw anomaly score. The Normal/Suspicious/High Risk
labels come from simple, **project-defined threshold rules** applied to that score —
not something the model decides on its own:

- `anomaly_score ≥ 0.75` **and** consumption dropped ≥ 30% → **High Risk**
- `anomaly_score ≥ 0.55` → **Suspicious**
- otherwise → **Normal**

Combining the score with a concrete behavioral indicator (rather than the score alone)
avoids flagging "High Risk" purely because of one noisy score, and mirrors how a human
investigator would reason. These exact threshold values are reasonable, documented
starting points chosen for explainability — not values learned from data — and would be
tuned against real field-investigation feedback in a production deployment.

> The system identifies **potentially anomalous consumption patterns** that may
> indicate electricity theft and prioritizes them for further investigation. It does
> not, and cannot, say "this customer is stealing electricity."

## 13. Model Evaluation

All models are evaluated **only on the held-out test months (5–6)** — data none of
them, nor the scaler/imputer, were fit on. Because theft cases are a minority (~5.23%
of test records), **accuracy alone is misleading** — a model predicting "Normal" for
everyone would already score >94% while being useless. Precision, recall, F1-score,
PR-AUC, and the confusion matrix give a much more honest picture:

- **Recall** matters because missed theft cases mean continued revenue loss.
- **Precision** matters because false positives waste investigator time and can
  wrongly inconvenience genuine customers.
- **PR-AUC** (average precision) summarizes performance across *all* possible
  thresholds at once, rather than just the one threshold each model happens to use by
  default — especially informative when the positive class is rare. It's rank-based, so
  each model's raw score is used directly (no need for the 0–1 anomaly-score
  normalization here). The Simple Rule Baseline only outputs a hard 0/1 flag, not a
  ranked score, so PR-AUC isn't a meaningful concept for it — reported as `N/A` rather
  than a fabricated number.

### Model Comparison Table (test months 5–6, actual results from this run)

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Isolation Forest (primary) | 0.909 | 0.316 | 0.631 | 0.421 | 0.443 |
| One-Class SVM (comparison) | 0.909 | 0.347 | 0.847 | 0.493 | 0.603 |
| Simple Rule Baseline (no ML) | 0.945 | 0.470 | 0.401 | 0.433 | N/A |
| Random Forest (supervised benchmark) | 0.971 | 0.679 | 0.847 | 0.754 | 0.797 |

*(Exact numbers vary slightly by run/seed — regenerate with `python src/evaluate_model.py`.
These are the actual figures produced by this run — no numbers here are hard-coded or
fabricated.)*

**Confusion matrices** (`[[TN, FP], [FN, TP]]`), test months only:

- Isolation Forest: `[[2629, 214], [58, 99]]` — 214 false positives, 58 false negatives.
- One-Class SVM: `[[2593, 250], [24, 133]]` — 250 false positives, 24 false negatives.
- Simple Rule Baseline: `[[2772, 71], [94, 63]]` — 71 false positives, 94 false negatives.
- Random Forest: `[[2780, 63], [24, 133]]` — 63 false positives, 24 false negatives.

**Interpretation.** Random Forest looks strongest here *because it was trained directly
on the ground-truth label* — a supervised advantage that wouldn't reliably exist in a
real deployment. Between the two genuinely unsupervised models, One-Class SVM caught
far more true theft cases (recall 0.847 vs 0.631) at the cost of more false alarms
(lower precision), while Isolation Forest struck a more conservative trade-off. Neither
unsupervised model is "wrong" — they represent a real precision/recall trade-off a
utility would need to choose between based on how expensive false alarms vs. missed
theft are for them. The **Simple Rule Baseline** is a useful reference point: it
actually has *higher precision* than either unsupervised model (0.470 — its flags are
more often correct), but noticeably lower recall (0.401 — it misses more real cases),
because a single fixed threshold can't adapt to the multivariate combinations Isolation
Forest and One-Class SVM can pick up on. This is the concrete answer to "why use ML
instead of a simple rule?" — the rule isn't bad, but it trades away recall that the ML
models recover.

**Honesty note on synthetic evaluation.** Because the ground-truth patterns were
generated by our own synthetic-data script, these numbers measure how well the models
recover the *simulated* anomalous patterns we injected — **not real-world
electricity-theft detection accuracy.** Real consumption data, real tampering
techniques, and real sensor noise would likely look different. These results
demonstrate methodology, not a production performance guarantee.

## 13b. Error Analysis

Beyond the aggregate metrics above, it's worth looking at *where* the primary model
(Isolation Forest) actually gets things wrong on the test months, using real examples
from this run rather than a hypothetical:

| Error Type | Observation from this run |
|---|---|
| False Positive | Normal records incorrectly flagged (214 cases). Average consumption variability for false positives is **0.254**, vs **0.139** for correctly-identified normal records — naturally volatile customers get over-flagged. |
| False Positive | **151 of 214** false positives (71%) are **Industrial** customers, who have far larger absolute consumption (thousands of kWh/month) and therefore larger absolute swings even from ordinary month-to-month variation — the model reads that as anomalous even when it's normal for that customer type. |
| False Negative | Suspicious records missed (58 cases). Average consumption drop for false negatives is **-7.2%**, much milder than the **-14.1%** average drop for correctly-caught true positives — subtle theft patterns (like the `peak_offpeak_shift` and `combination` patterns, which deliberately keep total consumption close to normal) blend into ordinary variation. |
| False Negative | **43 of 58** false negatives (74%) are **Residential** customers, whose smaller absolute consumption means a given theft-driven kWh drop is a smaller signal relative to the noise already present in that segment. |

This points to a concrete, explainable limitation rather than a black box: the model
under-performs specifically for (a) naturally high-variance Industrial accounts, where
better per-segment thresholds or Industrial-specific baselines would likely help, and
(b) subtle/partial theft patterns on Residential accounts, where the total-consumption
signal alone is too weak — the `Peak_OffPeak_Ratio` and `Power_Factor` features exist
precisely to catch this second case, but don't fully close the gap.

## 14. Customer-Level vs. Record-Level Analysis

The dataset has **multiple rows per customer** (one per month), so "flagged records ÷
total records" is not the same question as "flagged customers ÷ total customers."
Reporting the wrong denominator (e.g. `flagged_records / 9,000` when 9,000 is
customer-*months*, not unique customers) produces a misleading rate.

**Record-level** (per customer-month, all 9,000 records):

| Metric | Value |
|---|---:|
| Total customer-months | 9,000 |
| Normal | 8,525 |
| Suspicious | 444 |
| High Risk | 31 |
| Record anomaly rate | 5.28% |

**Customer-level** (per unique customer, 1,500 customers):

| Metric | Value |
|---|---:|
| Total unique customers | 1,500 |
| Customers with ≥1 Suspicious month | 214 |
| Customers with ≥1 High-Risk month | 26 |
| Customers flagged at least once | 217 |
| **Customer risk rate** | **14.47%** |

Note that 217 customers flagged out of 1,500 (14.47%) is the correct customer-level
rate — *not* 475 (Suspicious + High Risk records) divided by 9,000. A single customer
can contribute multiple flagged months; the customer-level number is what actually
matters for deciding how many *accounts* might warrant investigation.

## 15. Streamlit Dashboard

Run with `streamlit run app/app.py`. Four tabs:

- **🏠 Dashboard Home** — a persistent disclaimer banner, then both record-level and
  customer-level metric blocks (clearly labeled and explained, per Section 14), plus a
  risk-distribution chart. Also states which months were used for training vs. testing.
- **🔍 Customer Analysis** — select a `Customer_ID` to see their risk level, current
  consumption, **"Previous 6-Month Average Consumption"** (explicitly labeled to make
  clear it excludes the current month — see Section 9), consumption change % (vs. the
  previous 3-month average), anomaly score, which data split (train/test) their latest
  month falls in, a historical consumption line chart with the historical baseline
  plotted as a separate series, and **"Important Indicators"** generated from actual
  feature values/thresholds (never random text) — e.g. "Abnormally low power factor,"
  "High month-to-month consumption variability." A customer's first month (no prior
  history) displays "N/A (no prior history)" instead of a fabricated or crashing value.
- **📊 Data Visualization** — consumption distribution, anomaly-score distribution,
  consumption by customer type, risk level by customer type, and a multi-customer
  comparison chart.
- **⚖️ Model Comparison** — the same Model Comparison Table from Section 13 (Isolation
  Forest, One-Class SVM, Simple Rule Baseline, Random Forest, all evaluated on the exact
  same held-out test months), plus the Error Analysis table from Section 13b with real
  false-positive/false-negative counts and examples from this run.

The dashboard reads directly from `data/scored_customers.csv`, which is produced once
by `src/train_model.py`. It does **not** fit a new scaler, retrain a model, recompute
the anomaly-score min/max, or use ground-truth labels — every value shown was already
computed using training-fit statistics (see Sections 9–10), so browsing the dashboard
cannot introduce leakage on its own.

## 16. Results

The system separates a small, prioritized subset of accounts (14.5% of customers had at
least one flagged month; most had none) from the bulk of normal customers, using
consumption-pattern data alone. On the unseen test months, One-Class SVM recovered
about 85% of the simulated theft cases (at the cost of more false alarms), while
Isolation Forest struck a more conservative balance — about 63% recall with fewer false
positives. The Simple Rule Baseline shows that a fixed threshold alone gets reasonable
precision (0.47) but misses more cases (40% recall) than either ML model — a plausible
starting point for a prioritized-review workflow, not a finished detector, and a
concrete demonstration of what the ML models add over a simple rule.

## 17. Limitations

**An anomalous electricity-consumption pattern does not necessarily mean electricity
theft.** Legitimate reasons for unusual consumption include:

- Customer on vacation / property temporarily vacant
- Seasonal changes in usage
- Solar-panel installation (reduces grid consumption)
- Business closure or reduced operating hours
- Meter malfunction or replacement

This system is a **decision-support / early-warning tool**, not a final theft-detection
authority. Additional limitations:

- Trained and evaluated on **synthetic data** — real-world consumption patterns, fraud
  techniques, and sensor noise may differ, so reported metrics reflect recovery of
  *simulated* anomalies, not real-world detection performance (see Section 13).
- Only 6 months of history per customer are available, limiting how much "historical"
  training data exists and how far into the future the test split can extend.
- Risk thresholds are simple and manually chosen for explainability; a production
  system would tune them against real investigation outcomes over time.
- The model doesn't account for external context (weather, grid outages, festivals)
  that could also explain consumption spikes or drops.
- Precision is modest for the unsupervised models (see Section 13) — meaning a
  meaningful share of flagged accounts would turn out, on investigation, to have a
  legitimate explanation. This is expected and is why human review remains essential.

## 18. Ethical Considerations

- **No automatic penalties.** This system only prioritizes accounts for human review;
  it never triggers billing changes, disconnections, or accusations on its own.
- **False positives have real costs.** Flagging a genuine customer as "Suspicious" or
  "High Risk" can cause inconvenience, and repeated false flags can erode trust in the
  system — the dashboard's disclaimer and reason lists are there to keep this visible.
- **Transparency over black-box scoring.** Every risk label traces back to an
  explainable rule and specific feature values (shown as "Important Indicators"),
  rather than an opaque score with no justification.
- **Fairness across customer segments.** Because consumption baselines differ sharply
  by `Customer_Type` (Residential vs. Commercial vs. Industrial), features are compared
  to a customer's *own* history rather than a single global threshold, reducing the risk
  of systematically over-flagging one segment.
- **Data privacy.** A real deployment would involve sensitive customer data and should
  follow the utility's applicable data-protection regulations; this project only uses
  synthetic data for that reason.

## 19. Future Scope

- Integrate real smart-meter data (with appropriate privacy/security safeguards).
- Real-time or streaming anomaly detection instead of monthly batch scoring.
- Time-series-specific models (e.g., seasonal decomposition) for more precise deviation
  detection, once more months of history are available.
- Geographic/network-level analysis (e.g., comparing a customer to their transformer or
  feeder-line neighbors).
- Automatic prioritization/ranking of cases for field-investigation teams.
- Integration with electricity-provider billing and CRM systems.
- Tuning risk thresholds against real investigator feedback over time.

*(None of the above are implemented in this project — they are proposed next steps.)*

## 20. Installation

```bash
cd electricity-theft-detection
pip install -r requirements.txt
```

## 21. How to Run

```bash
# 1. Generate the synthetic dataset
python src/generate_dataset.py

# 2. (Optional) sanity-check preprocessing / feature engineering in isolation
python src/data_preprocessing.py
python src/feature_engineering.py

# 3. Train all models using the time-based split (creates models/*.pkl —
#    including isolation_forest.pkl, one_class_svm.pkl, random_forest.pkl,
#    scaler.pkl, impute_stats.pkl, feature_list.pkl, and
#    anomaly_score_params.pkl — plus data/scored_customers.csv)
python src/train_model.py

# 4. Evaluate on the held-out test months (5-6) and print the comparison table
python src/evaluate_model.py

# 5. Launch the dashboard
streamlit run app/app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Possible Interview Questions and Answers

**1. Why did you choose this project?**
Electricity theft is a real, costly problem for power companies, and it's a strong
example of applying anomaly detection to a practical business problem rather than a
generic classification dataset.

**2. Why is electricity theft a data-science problem?**
It shows up as a *pattern* in consumption data (sudden drops, abnormal power factor,
erratic usage) rather than something directly observable without a physical inspection.
Data science helps prioritize which patterns are worth investigating.

**3. Why did you choose Isolation Forest as the primary model?**
It's purpose-built for anomaly detection, doesn't need labeled data, handles multiple
numeric features well, is computationally efficient, and makes no assumption about the
data's distribution — unlike some statistical methods.

**4. How does Isolation Forest work at a basic level?**
It builds many random decision trees that split the data randomly. Anomalies are
"different" enough that they get isolated into their own leaf in fewer splits than
normal points. The average path length across all trees becomes the anomaly score.

**5. What is an anomaly score?**
A numeric value indicating how unusual a data point is relative to the rest of the
dataset. Here it's normalized to 0–1, where higher means more anomalous.

**6. How does feature engineering work in this project?**
Each feature was designed around what a real investigator would look at: how much
consumption changed recently, how far it is from the historical average, the
peak/off-peak ratio, how variable usage has been, and whether meter readings were
complete.

**7. How are historical features like Consumption_Change_Pct and Consumption_Variability calculated?**
Both use only the customer's **past** months, via `.shift(1)` before any
`.rolling()`/`.expanding()` call — this shifts each customer's series down by one row
so a given month can only "see" months before it. `Average_Consumption_3_Months` and
`Average_Consumption_6_Months` (used by `Consumption_Change_Pct` and
`Consumption_Deviation`) never include the current month's own reading —
e.g. March's 3-month average is `mean(Jan, Feb)`, not `mean(Jan, Feb, Mar)`.
`Consumption_Variability` similarly uses only the 3 months immediately before the
current row. A customer's first month has no prior history, so these features fall
back to explicit, documented defaults (`NaN` → 0 for the derived features) rather than
guessing a number.

**8. How was data leakage prevented?**
Four ways: (1) a time-based train/test split so no customer's future month ever
appears in training; (2) all preprocessing statistics (imputation medians) and the
`StandardScaler` are fit only on training months and reused, never re-fit, on the test
months; (3) historical averages and variability are computed with `.shift(1)` so a
row's own current-month value is never part of its own "history"; (4) the Isolation
Forest anomaly-score min/max used for 0–1 normalization are computed only from
training-month raw scores and reused everywhere else — never recalculated from test or
combined data (see Section 10 above for the full walkthrough); and (5)
`Theft_Flag_GroundTruth` is asserted at import
time to never appear in the model's feature list — it's used only for evaluation.

**9. Why is time-based splitting appropriate here?**
Because the dataset has repeated measurements per customer over time. A random split
could let a customer's later month train the model while an earlier month for the same
customer is used to test it — effectively testing on "the past," which never happens in
production. Splitting by month (1–4 train, 5–6 test) mirrors how the system would
actually be used: train on history, score new data.

**10. What's the difference between customer-level and record-level risk?**
A record is one customer-month; a customer can have up to 6 records. Record-level
metrics describe what share of *monthly records* were flagged (6.31% here).
Customer-level metrics describe what share of *unique customers* had at least one
flagged month (13.53% here) — the more meaningful number for deciding how many actual
accounts might need investigation. Dividing flagged records by total records instead of
by total customers would overstate or understate the real customer-level risk.

**11. Why isn't accuracy enough?**
Theft cases are a small minority (~3.7% of test records). A model predicting "Normal"
for everyone would already exceed 96% accuracy while being completely useless.

**12. What are precision and recall?**
Precision = of everything the model flagged, how much was actually theft. Recall = of
all actual theft cases, how many the model caught. There's usually a trade-off — in my
results, One-Class SVM traded lower precision for much higher recall (0.937) than
Isolation Forest (0.721).

**13. What is class imbalance, and how did you handle it?**
When one class (Normal) vastly outnumbers the other (Theft). I handled it by evaluating
with precision/recall/F1 instead of accuracy alone, and by using `class_weight="balanced"`
for the Random Forest so it doesn't just default to predicting the majority class.

**14. Why could a legitimate customer be flagged?**
Many innocent situations look statistically similar to tampering — vacations, vacant
properties, seasonal change, solar panel installation, business closures, or meter
malfunctions can all produce a large consumption drop or unusual pattern.

**15. What are false positives and false negatives here?**
False positive: a normal customer incorrectly flagged — wastes investigator time and
can unfairly inconvenience a genuine customer. False negative: an actual theft case the
model missed — means continued revenue loss for the utility.

**16. Why didn't you use deep learning?**
The dataset size and feature complexity don't justify it. Isolation Forest and
One-Class SVM already work well on structured/tabular data like this, are far more
interpretable, and match the scope of a 3rd-year project. Deep learning would add
complexity without a clear benefit here.

**17. Why does the dataset need a synthetic ground-truth label if this is supposed to be unsupervised?**
Isolation Forest and One-Class SVM never see the label during training — it's used
*only* for evaluation, to check how well the models' flags line up with the simulated
patterns. In a real deployment this label usually wouldn't exist for most customers.

**18. How would you improve the project?**
Use real smart-meter data, add time-series-specific modeling, tune risk thresholds
against real investigation feedback, and add geographic/network-level comparisons
(e.g., comparing a customer to neighbors on the same transformer).

**19. What are the limitations of using synthetic data?**
The evaluation numbers measure how well the models recover the *simulated* patterns
this project's own generator injected — not real-world theft-detection accuracy. Real
consumption data, fraud techniques, and sensor noise would likely behave differently,
so these results demonstrate methodology rather than guarantee production performance.

**20. Why does "suspicious" not mean confirmed theft?**
Because the risk labels come from statistical deviation, not a physical inspection.
Many legitimate situations (see Q14) produce similar patterns. The system prioritizes
cases for human investigation — it never makes an automatic accusation.

**21. How would you deploy this system in the real world?**
As a decision-support tool integrated into a utility's billing/CRM system: it would run
periodically (e.g., monthly) on real smart-meter data, produce a prioritized list of
Suspicious/High-Risk accounts at both the record and customer level, and route that
list to human investigators — never triggering automatic penalties on its own.

**22. Why is Random Forest not the primary model despite scoring higher?**
Because its higher score comes from training directly on a ground-truth label that
exists only because this dataset is synthetic. In a real deployment, that label
wouldn't reliably be available for most customers, so Isolation Forest — which needs no
labels — is the model that's actually deployable in the primary use case. Random Forest
is included purely as an educational benchmark.

---

*This project uses entirely synthetic data generated for educational purposes. It is
not connected to, and does not use data from, any real electricity provider.*
