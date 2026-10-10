# Glucose CSV cleanup and EDA

## Personalized meal-response model (recommended)

```powershell
python backend/regression-ml-glucose/personalized_model.py
python backend/regression-ml-glucose/test_personalized.py -v
```

This is separate from the older daily-average `model.py`. It predicts glucose at
30, 60, 90 and 120 minutes after a logged meal, by predicting change from baseline
and adding observed pre-meal CGM. Features are current logged food and pre-meal
glucose trend, clock time, and prior-24-hour treatment logs. Nutrition remains
keyword/quantity features, not inferred carbohydrate estimates. Meal information
is assumed available when predicting; this retrospective evaluation is conditional
on that assumption. No future treatment or glucose enters an input.

For EACH evaluated patient, a separate population Ridge model is trained on
OTHER patients' completed first-seven-day meal outcomes. The held-out patient's
own outcomes never enter that population fit or its imputer/scaler. For day 8,
a patient-specific regularized residual model and shrunken offset use that
recording's days 1–7. For day 9, it refits on days 2–8, and so on. Only response
labels completed strictly before the day's cutoff are eligible for adaptation.
The adapter is frozen for that day, so earlier same-day outcomes cannot leak into
later forecasts. Histories never cross recordings. Fewer than five usable history
meals triggers population-only fallback. Alpha=100 and shrinkage=10 are fixed
development choices, not tuned on these held-out outcomes. This is leave-one-
patient-out evaluation, rather than the daily model's 60/20/20 partition.

Before standardizing, inference features are bounded to the population training
minimum/maximum using a fitted clipping scaler, preventing unbounded linear
extrapolation for unseen treatment/quantity values. This does not edit raw data
or guarantee plausibility; external validation and out-of-distribution flags
remain necessary. The personal adapter uses the same bounded feature space.

Meal examples require at least four pre-meal readings within 30 minutes, a latest
baseline no older than five minutes, and valid bracketed outcome readings within
15-minute gaps. Any second logged meal in the next 120 minutes excludes the
episode. That future-log rule defines a retrospective isolated-meal evaluation
cohort; deployment cannot know a future snack unless supplied as a plan.
Targets may be interpolated; no source provenance flag is available in the
prepared ten-column data to distinguish already-interpolated source readings.

Results under `backend/artifacts/personalized-meal-regression/`:

- `meal_predictions.csv`: each patient/meal/horizon prediction from day 8 onward,
  pooled prediction, observed-baseline prediction, actual glucose, personal
  history size and cutoff, and population-only flag.
- `day8_predictions.csv`: just day 8 meal forecasts (not daily averages).
- `patient_metrics.csv`: per-patient, per-horizon R², MAE, RMSE, pooled-model MAE,
  and persistence-baseline MAE. R² is undefined for a single meal and can be
  unstable for small samples. Errors are mg/dL, not accuracy percentages.
- `meal_eligibility.csv`: logged/usable meals and duration for every recording,
  including those without day 8 evaluations.
- `population_training_audit.csv`: other-patient IDs used for each held-out model.
- `feature_contributions.csv`: exact additive contributions to predicted glucose
  CHANGE, from the combined population and personal linear coefficients relative
  to standardized zero (population-imputed/scaled reference).
- `patient_feature_importance.csv`: each patient's mean absolute/signed response
  contributions per horizon. The direct observed-glucose baseline is added outside
  these response contributions. Rankings explain predictions, not causal effects;
  correlated inputs and sparse food/treatment variation limit interpretation.
- `personalized_models.joblib`: population pipeline and individual coefficients,
  offset, feature schema and cutoff per patient/recording/forecast day. The
  `predict` function accepts these components and a prepared meal-feature frame;
  add pre-meal baseline to its returned changes. Load trusted artifacts only.
- `metrics.json`: evaluation count and pooled/personalized/persistence MAE per
  horizon. Outcomes overlap across horizons/days, so rows are not independent.

No calibrated uncertainty interval is produced yet. Sparse-history fallback and
meal counts are reported. Future work should use an independent development cohort
to tune personalization and calibrate intervals before final external evaluation.

Run `python backend/regression-ml-glucose/eda.py` from the repository root.

Creates `backend/data/5min-glucose-data-cleaned.csv` with exactly the ten requested
columns, preserving source rows and units. Blank dietary entries are replaced
with `No food ingested` under the user's modeling assumption. Other values and
missing entries are preserved. Insulin dose
means the source's subcutaneous `Insulin dose - s.c.` column; intravenous dose
is excluded. The original CSV is unchanged.

EDA outputs are in `backend/reports/regression-ml-glucose/`: first rows, shape,
patient/recording counts, date range, duplicates, missingness, per-patient coverage,
numeric count/mean/sample standard deviation/minimum/quartiles/maximum, and
categorical summaries. Inspect `eda_report.txt` for the combined report.
Medication and dietary text are not coerced to numeric doses. Empty glucose
readings remain missing, rather than becoming zero. Numeric parsing is for EDA
only; the cleaned CSV retains other source values.

The original CGMacros model was moved to `legacy_cgmacros_model.py` for reference.
The active `model.py` here uses this uploaded dataset and Ridge linear regression.

## Food features for regression

Run `python backend/regression-ml-glucose/prepare_data.py` to create a separate
`backend/data/5min-glucose-data-prepared.csv`. Unknown food entries (including
`data not available`) are written as `NA`; the original and cleaned CSVs stay intact.
By project assumption, absent food entries mean no food intake at that timestamp.
The `explicit_no_food` indicator includes these assumed no-food rows as well as
explicit no-food text. No nutrition quantities are invented.

Features distinguish logged food, unknown entries, and explicit no-food entries.
Counts of food log entries in 0–30, 30–60, and 60–120 minute bands capture delayed
associations without copying meal descriptions onto every glucose row. These are
log counts, not confirmed counts of separate meals. History is isolated by patient
and recording, uses only current/past timestamps, and flags the first two hours
as incomplete history. Minutes since the last logged food remains missing before
the first log, with a separate indicator. Unlogged intake remains unobservable.

For linear regression, use numeric food indicators and time bands alongside
pre-prediction glucose, time of day, and separately prepared treatment covariates.
Impute/scale using training data only; never encode food descriptions as arbitrary
integers. Patient and recording IDs are grouping keys. For spike prediction,
construct a future glucose-change target and split chronologically with a gap at
least as long as the prediction horizon. Do not use future glucose as an input.
Food coefficients describe association; they do not establish causation. This
script prepares features only; run the model separately as described below.

## Training the day 8 model

```powershell
python backend/regression-ml-glucose/prepare_data.py
python backend/regression-ml-glucose/model.py
python backend/regression-ml-glucose/test_model.py -v
```

Food features include overlapping ingredient groups, logged food weight in grams
and liquid volume in ml (NOT carbohydrate or calorie estimates), quantities that
could not be parsed, and recent food-log counts. Group matching is a transparent
keyword heuristic; inspect `food_mapping.csv` for unmatched/ambiguous descriptions.
The grams/ml totals describe logged quantities, not measured nutrients.

Medications are represented by identity indicators and separate named-drug doses
in mg (metformin g is converted to mg). Ambiguous combination descriptions and
non-metformin g entries are flagged rather than silently corrected. Insulin doses
in IU are summed within the log; insulin names and premix status are retained.
Pump rates remain in IU/hour, distinct from bolus IU. Non-numeric pump entries
are flagged; missing rates are never forward-filled. No logged medication entry
is encoded as zero **logged** dose, not proof that no drug was active. Source
text remains unchanged; `parsing_review.csv` lists entries needing manual review.
Mapping and review files are in `backend/reports/regression-ml-glucose/`.

The target is **next-day average CGM**, not a two-hour meal response or a full
day's glucose curve. Each day is a 24-hour block anchored at the recording's
first timestamp, avoiding a partial calendar day 1. Each of seven history days
and the target day needs at least 230 CGM readings (~80% of 288), and must fit
within the recording duration. Windows never span recordings or patients.
There is no interpolation or future food/medication input. Daily history glucose
mean/std/min/max and seven-day averages of food/treatment features form the inputs.

Patients are split 60% train / 20% validation / 20% test (seed 42). All overlapping
windows for a patient stay in that patient's partition. All three partitions use
every eligible rolling window: days 1–7 predict 8, 2–8 predict 9, 3–9 predict 10.
Thus these evaluation sets test a shared model on separate patients with seven
days of their observed history as features; no separate patient model is fitted.
Ordinary linear regression and Ridge alpha candidates are compared on validation
only. CBG and available no-food/history indicators also enter the historical
features. IDs, raw text, target-day CGM, and future factors never enter as features.
Imputation and scaling fit train only.
Test data is not used for parameter selection or refitting.

Outputs under `backend/artifacts/regression-ml-glucose/`:

- `train.csv`, `validation.csv`, `test.csv`: evaluated feature/label rows and predictions.
- `split_manifest.csv`: window assignments (patient partitions are disjoint).
- `day8_predictions.csv`: first-seven-days predictions where day 8 is eligible,
  with split and actual outcome. Train rows are in-sample, validation selected alpha;
  only test rows provide untouched performance estimates.
- `recording_eligibility.csv`: eligibility and excluded recordings.
- `metrics.json`: MAE/RMSE and day-7/seven-day-mean baseline errors, all mg/dL.
- `rolling_predictions.csv`: predictions for all eligible forecast days, with split.
- `patient_metrics.csv`: MAE, RMSE, baseline errors, and R² by patient.
- `coefficients.csv`: coefficients for standardized features, conditional on others;
  correlated features and shrinkage limit interpretation. These are not causal effects.
- `linear_regression.joblib`: fitted pipeline and feature schema.

Historical CBG measurements enter the model. Suspicious source values
are preserved for review; no unverified correction or nutrient estimation is made.
Prediction uncertainty and meal-spike forecasting remain separate future tasks.

The command prints R², MAE, and RMSE for all three partitions in the terminal.
R² is not a classification accuracy percentage; it can be negative. Forecasts
are next-day means, not five-minute CGM trajectories. The model remains shared
across patients and fixed during evaluation; it is not retrained after each day.
The CLI uses `backend/data/5min-glucose-data-prepared.csv` when present, otherwise
the uploaded copy under `backend/copies of old data/`. Override with `--data`.
