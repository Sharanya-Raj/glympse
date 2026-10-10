# Glucose CSV cleanup and EDA

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
windows for a patient stay in that patient's partition. Training can use multiple
seven-day windows; validation/test use the earliest eligible window per recording.
Thus these evaluation sets test a shared model on separate patients with seven
days of their observed history as features; no separate patient model is fitted.
Ridge alpha is selected on validation only. Imputation and scaling fit train only.
Test data is not used for parameter selection or refitting.

Outputs under `backend/artifacts/regression-ml-glucose/`:

- `train.csv`, `validation.csv`, `test.csv`: evaluated feature/label rows and predictions.
- `split_manifest.csv`: window assignments (patient partitions are disjoint).
- `day8_predictions.csv`: first-seven-days predictions where day 8 is eligible,
  with split and actual outcome. Train rows are in-sample, validation selected alpha;
  only test rows provide untouched performance estimates.
- `recording_eligibility.csv`: eligibility and excluded recordings.
- `metrics.json`: MAE/RMSE and day-7/seven-day-mean baseline errors, all mg/dL.
- `coefficients.csv`: coefficients for standardized features, conditional on others;
  correlated features and shrinkage limit interpretation. These are not causal effects.
- `linear_regression.joblib`: fitted pipeline and feature schema.

The retained CBG measurements do not enter this model. Suspicious source values
are preserved for review; no unverified correction or nutrient estimation is made.
Prediction uncertainty and meal-spike forecasting remain separate future tasks.
