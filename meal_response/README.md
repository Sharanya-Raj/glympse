# Meal-response training

## Side-by-side backend comparison

Open `meal_response/model_comparison.ipynb` for executed comparisons of the
adapted backend, meal_response linear regression, and meal_response XGBoost on
CGMacros and Shanghai T2DM, both separately and pooled. It includes identical
held-out-meal comparisons, patient-balanced errors, paired bootstrap intervals,
full-curve results, and global/individual feature contributions. The Sharanya
daily-mean task is reported separately from the two-hour meal task.

To recreate the comparison and notebook from the repository root:

```powershell
uv run python -m meal_response.comparison_notebook
```

This verifies the saved pooled run and trains source-only models with the full
default tuning budget. Artifacts are saved in a new `artifacts/model_comparison/`
run. To regenerate only the executed notebook from a prepared run:

```powershell
uv run python -m meal_response.comparison_notebook --run artifacts/model_comparison/RUN_ID
```

The notebook uses saved artifacts and does not retrain when you run its cells.
Select this repository's `.venv` Python kernel in your notebook editor. The
comparison builder uses IPython (available through the local ipykernel install).
The backend adapter intentionally uses the shared patient split and supplied
curve labels; the notebook documents feature and target changes from the
original backend protocol. These are exploratory comparisons on an inspected
test set.

Train ordinary least-squares linear regression and XGBoost on CGMacros, Shanghai
T2DM, or both. Each predicts a 24-point, five-minute glucose curve for two hours
after the planned meal. The curve determines 30/60/90/120-minute glucose, peak,
rise from baseline, earliest time to peak, positive incremental AUC, and signed
net incremental AUC.

Run from the repository root with the existing environment:

```powershell
.venv/Scripts/python.exe -m meal_response audit
.venv/Scripts/python.exe -m meal_response compare
.venv/Scripts/python.exe -m meal_response compare --include-enhanced
.venv/Scripts/python.exe -m meal_response train --dataset cg_macros --model both
.venv/Scripts/python.exe -m meal_response train --dataset t2dm --model both
.venv/Scripts/python.exe -m meal_response train --dataset combined --model both
.venv/Scripts/python.exe -m unittest discover -s meal_response/tests -v
```

All dependencies are already in the project environment: pandas, NumPy,
scikit-learn, XGBoost, joblib (through scikit-learn), and matplotlib. No nutrition
API calls are made. Default inputs are `data/T2DM_5min_with_macros.csv` and
`data/cg_macros`; override with `--t2dm PATH` and `--cg-root PATH`.

When both sources exist they are audited together, even for a single-source run.
A missing unselected source is allowed; a missing selected source fails that
experiment. The master split is based on core-eligible patients and reused for
enhanced comparisons. Seeded assignments remain stable while each source's
eligible identities are unchanged.

Every command creates a timestamped directory under `artifacts/meal_response`
(override the parent with `--output`). Outputs include input fingerprints,
CGMacros file inventory, per-patient exclusion reports, feature examples, and
patient split manifests. Each trained model adds `model.joblib`,
`feature_schema.json`, `config.json`, `metrics.json`, `test_predictions.csv`, and
an evaluation plot. Comparisons add a CSV/JSON table and comparison plot.
Unsupported configurations get failure reports and a nonzero command exit code.

## Protocol

- Inputs are the planned meal's carbohydrate, protein, fat, calories, optional
  fiber, time of day, recent 60-minute CGM, and summaries of up to seven preceding
  days. No static demographics or labs, cohort, patient/source ID, future food,
  future medication, or future activity enters a model.
- Short histories are allowed. Baseline must be no older than five minutes;
  at least 10 positions on the 13-point recent CGM grid must be usable.
- Required nutrition must be finite and nonnegative. T2DM requires
  `macro_status=ok`; partial/unresolved estimates are excluded. Missing fiber
  gets a missingness indicator. Historical nutrition totals reflect logged
  complete meals and include separate logging/coverage indicators.
- Exclude meals with any other logged food event in the next two hours,
  including food without nutrition estimates. Forecasts assume no additional
  food in that period; unlogged food remains a limitation.
- CGMacros actual `Amount Consumed` is audited, never used as a predictor or
  post-meal adjustment to planned macros. Actual intake may differ from the
  planned/recorded portion.
- All 24 outcome readings must exist within 2.5 minutes of their target times.
  The pipeline does not create new interpolated labels or bridge recording gaps.
- Split approximately 70/15/15 by **patient**, stratified by source/cohort where
  possible. Visits for one person stay together. Cohorts with fewer than three
  eligible patients stay in training. Each experiment requires nonempty train,
  validation, and test partitions.
- Train-only imputation/scaling and medication vocabulary prevent fitting on
  held-out patients. Combined training gives sources equal total weight and
  patients equal weight within source. Validation tunes XGBoost; no refitting
  on validation occurs after selection.
- Curve RMSE averages each patient's curve RMSE. Endpoint MAE/RMSE/R² use
  patient-balanced observation weights. Confidence intervals bootstrap patients,
  not meal rows. Reports include source, cohort, and history-length breakdowns
  and a constant pre-meal-glucose persistence baseline.

## Provenance and necessary adjustments to the plan

The supplied T2DM file has 336,623 rows, with 224,336 marked interpolated. Of
3,060 nutrition-linked meals, 575 have complete nutrition estimates. After all
exclusions, the initial audit found 503 meals from 62 patients; CGMacros yielded
1,398 meals from 45 patients.

For T2DM **inputs**, discard flagged interpolation and reconstruct the recent
five-minute grid from original readings already available by meal time. Original
Shanghai CGM is generally 15-minute data, so allow a 15-minute bracketing gap
rather than the proposed ten-minute gap. Hold the latest reading for at most
five minutes. Historical statistics use only unflagged measurements.

CGMacros local exports populate Dexcom densely every minute without original
observation flags. Use history ending five minutes before the meal as a
conservative guard against interpolation using a future five-minute endpoint.
Baseline is consequently five minutes old. This assumes interpolation between
adjacent original Dexcom readings; original provenance cannot be established
from these files. The audit and model reports state that limitation. CGMacros
`METs` is physical activity multiplied by ten, not medication; divide by ten.

Default `--label-policy supplied` uses provided outcome curves, including their
existing interpolation. Reports show T2DM interpolated-label fractions; these
results **do not establish accuracy against independently observed five-minute
glucose measurements**. `--label-policy observed` refuses CGMacros targets
because originals cannot be identified and removes flagged T2DM targets. The
24-point requirement normally makes observed-only T2DM unsupported too. The
audit records exclusions; the policy is never silently relaxed.

Enhanced experiments are source-specific: CGMacros adds past HR, METs, and
activity calories; T2DM adds logged drug/route/unit-specific recorded doses and
counts plus unknown-history indicators. Preserve g, mg, IU, and IU/H separately.
Basal-rate sums describe recorded rates, not inferred insulin exposure. Empty
medication logs do not establish that no drugs were taken. Unparseable records
are retained and counted; no medication effects are interpreted causally.

## Confidence and larger glucose-rise weighting

Training now starts with the source/patient balancing weights and multiplies
them by meal quality and a mild large-rise factor, followed by one overall
mean-1 rescaling. Confidence is weighting metadata, not a prediction input.

- T2DM's `macro_confidence` is read from the supplied CSV. Confidence >=0.9
  keeps the normal factor of 1. Below 0.9 the factor is confidence/0.9; e.g.
  0.45 receives 0.5. Zero-confidence meals contribute no loss. Missing T2DM
  confidence receives 0.5. Invalid nonmissing values outside [0,1] fail training.
- CGMacros has no recorded macro-confidence field. Its missing confidence
  stays explicitly unscored and receives factor 1; a score of 1 is not fabricated.
- Training meals with positive peak-minus-baseline rise above their source's
  training-only 75th percentile receive an extra factor of 1.25. This emphasizes
  larger responses without adding held-out meals to fitting. Combined confidence
  and rise factors still reduce uncertain meals: 0.45 confidence and a large
  rise together yield 0.5 * 1.25 = 0.625 of normal base influence.
- XGBoost validation early stopping uses confidence weights but no rise boost.
  Candidate selection and reported test metrics retain the original unweighted
  patient-balanced protocol. Test meals never determine a boost threshold.
- Partial/incomplete T2DM nutrition estimates remain excluded; their reported
  totals omit unresolved foods and cannot safely become complete meals just by
  assigning a lower weight.

Both linear regression and XGBoost use these weights for all dataset options.
Configure them with `--high-confidence-threshold`,
`--missing-confidence-factor`, `--large-rise-multiplier`, and `--rise-quantile`.
Set `--large-rise-multiplier 1` to disable extra rise emphasis. Each model saves
`training_weights.csv`, `validation_weights.csv`, and `weighting.json`, including
raw confidence, separate multipliers, final weights, and fitted rise thresholds.
Patient/source balance is the starting point; quality weighting intentionally
changes their final contribution and is not renormalized within each patient.

The plotted "Meal 1" is a held-out meal. Weighting a response observed in a test
plot is implemented as an exploratory training-policy change, not a promise of
improving that particular meal. Comparisons using the same inspected test set
are exploratory; an independent cohort is needed for a fresh final evaluation.

## Resource options

Defaults: 30 seeded XGBoost trials, up to 2,000 trees, 50-round early stopping,
300 bootstrap resamples, seed 42, one XGBoost worker. Use `--trials`,
`--max-trees`, `--bootstraps`, and `--jobs` to change them. Each trial fits 24
regressors with shared sampled parameters and horizon-specific early stopping.
A plumbing check can use reduced settings:

```powershell
.venv/Scripts/python.exe -m meal_response compare --trials 1 --max-trees 50 --bootstraps 10
```

After training, verify input fingerprints, patient isolation, current feature
construction, saved predictions, derived outcomes, and online inference against
each completed model (and optionally regenerate its plots):

```powershell
.venv/Scripts/python.exe -m meal_response.verify_artifacts artifacts/meal_response/RUN_ID --refresh-plots
```

Verification writes `verification.json` alongside the run results. It is
intended for completed supplied-label runs, including enhanced comparisons.

## Inference

Python: `meal_response.forecast(bundle, patient, meal_time, planned_meal)`.
`bundle` is loaded with joblib; `Patient` is defined in `meal_response.data`.
Inference reuses training feature construction and truncates histories to the
preceding seven days. Load only trusted joblib artifacts.

```powershell
.venv/Scripts/python.exe -m meal_response predict --model-file PATH/model.joblib --input request.json --output forecast.json
```

Example `request.json` with a minimal hour of original T2DM readings:

```json
{
  "source": "t2dm",
  "meal_time": "2026-10-10 12:00:00",
  "planned_meal": {"carbs": 50, "protein": 25, "fat": 15, "calories": 435},
  "glucose_history": [
    {"time": "2026-10-10 11:00:00", "glucose": 110, "interpolated": false},
    {"time": "2026-10-10 11:15:00", "glucose": 112, "interpolated": false},
    {"time": "2026-10-10 11:30:00", "glucose": 111, "interpolated": false},
    {"time": "2026-10-10 11:45:00", "glucose": 109, "interpolated": false},
    {"time": "2026-10-10 12:00:00", "glucose": 108, "interpolated": false}
  ],
  "meal_history": [],
  "medication_history": [],
  "activity_history": []
}
```

Optional meal-history fields: `time`, `carbs`, `protein`, `fat`, `calories`,
`fiber`, boolean `nutrition_complete`. Activity: `time`, `mets` (already divided
by ten), `hr`, `activity_calories` (per-minute kcal). Medication: `time`, `drug`,
`route`, `unit`, `dose`, `raw`, boolean `parsed`. Drug spelling must match the
training adapter; use `medication_events` to normalize source-style records.
CGMacros inputs use its dense supplied glucose representation and the same
five-minute history guard. Use a single consistent local timeline; mixed naive
and timezone-aware timestamps are unsupported.

Output includes the curve, baseline, outcomes, available history, and
no-additional-food assumption. Positive iAUC clips negative increments before
trapezoids; `net_iauc` retains them. Both include zero increment at minute zero.
Peak excludes baseline and considers minutes 5–120; peak rise can be negative.
Time-to-peak uses the earliest tied point. No app integration or deployment is
included.
