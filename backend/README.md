# Backend

Current email/password sign-in, password recovery, and patient 2088 import:
[EMAIL_AUTH.md](EMAIL_AUTH.md). The legacy local example accounts described below
are retired; provision users in Cognito.

The maintained ML pipeline lives in `ml/`.
Read `ml/README.md` for account creation, uploads, prediction, daily feature
rankings, tests and benchmarks. `app.py` is the authenticated API;
`lambda_function.py` is the existing deployment entry point.

Install ML dependencies with `python -m pip install -r backend/requirements.txt`.
API runtime and test dependencies are included in this list.

## Local patient examples

The frontend and API use the two workbooks in `data/frontend_test/` as example
accounts. Add these settings to your gitignored `backend/.env`:

```ini
GLYIMPSE_ENV=dev
GLYIMPSE_SEED_RECORDINGS=2000_0_20201230,2001_0_20201102
GLYIMPSE_SEED_PASSWORD=patient-demo-123
```

Run `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` from the
repository root and open <http://127.0.0.1:8000>. The sign-in screen offers
`patient2000` and `patient2001`; both use `patient-demo-123` with the configuration
above. Each account has its own recorded meals, treatments, and glucose history.
The existing converter interpolates CGM to five minutes only across gaps of at
most 30 minutes; meal and treatment timestamps stay intact. Replay starts on day 8.
Existing account histories and forecasts survive server restarts.

`GLYIMPSE_DATA_DIR` overrides the default `data/frontend_test/` source directory.
Example accounts require development mode; disable the seed settings in production.
The shared prediction model still comes from `backend/artifacts/population.joblib`.

# Glucose forecasting modules

`data.py`, `eda.py`, `features.py`, `model.py`, and `explain.py` provide the daily
forecasting pipeline used by `meal_response/comparison_notebook.py`.

## Sharanya prepared forecasts (default)

From the repository root, run:

```powershell
uv run python -m backend.eda
uv run python -m backend.model
uv run python -m backend.explain
```

The default input is `data/sharanya_cg/`, containing `train.csv`,
`validation.csv`, and `test.csv`. Override it with `--data PATH`.
These are prepared daily mean-glucose forecasts, so the backend uses their
`target` and supplied patient-disjoint splits rather than building meal responses.
Only `day1_glucose_*` through `day7_glucose_*` and `history_*` columns are predictors.
Patient IDs, recording IDs, target day, split, existing predictions, and errors
are excluded. Imputation and model fitting use training rows only.

The model saves `regression.joblib`, `metrics.json`, `validation_predictions.csv`,
and `test_predictions.csv` under `backend/artifacts/`. EDA saves reports under
`backend/reports/`; SHAP explains test rows by default (`--split validation`
selects validation rows). The model predicts daily mean glucose, not a meal curve.

## Raw CGMacros data

```powershell
python -m pip install -r backend/requirements.txt
python -m backend.eda --data backend/data
python -m backend.model --data backend/data
python -m backend.explain
```

Download and extract the real participant data from
https://physionet.org/content/cgmacros/1.0.0/ into `backend/data/`.
The loader searches recursively for `CGMacros-*.csv`. Data and generated results
are excluded from Git.

## Modules

- `data.py`: schema validation, timestamps, numeric conversion, participant IDs.
- `eda.py`: missingness, participant coverage, daily glucose summaries, usable
  meal counts, feature/target summaries, and extracted meal table (CSV reports).
- `features.py`: pre-meal features and timestamp-aligned two-hour labels.
- `model.py`: XGBoost multi-output regression, day 8 evaluation, saved pipeline.
- `explain.py`: day 8 SHAP contributions and feature rankings for a chosen target.

## Adaptation rationale

Reviewed https://github.com/PSI-TAMU/CGMacros/blob/main/parse_data.ipynb.
It already trains XGBoost regression for breakfast AUC/iAUC with leave-one-person-out
evaluation. This implementation is newly written rather than a copied notebook:
it adopts the model family but changes the targets and evaluation protocol.
It avoids deprecated DataFrame.append, row-stride time assumptions, repeated macro
unit conversion, and demographic matching by positional array index. Macros stay
in grams. Demographics, photos, and medications are not model inputs in this version.

## Protocol and limits

Training uses calendar study days 1–7; testing uses day 8 for known participants.
Days 9–10 are unused. This is an initial pooled model with personal glucose-history
features, not a separately calibrated personal model or a new-person evaluation.
Do not tune on day 8; add development-person validation before hyperparameter search.

Features use only the meal information and readings at/before the meal. Assume
the entered macros describe the planned meal; actual amount consumed is excluded.
Sensor selection is explicit (Dexcom by default); sensors are never silently mixed.
Targets interpolate to 30/60/90/120 minutes with a maximum 15-minute bracketing gap.
Baseline must be observed within 10 minutes before the meal. Meals with another
meal in the following two hours or incomplete target coverage are excluded.
Peak, time-to-peak, and excursion labels use a five-minute grid over 0–120 minutes.
Calendar day 1 may be partial; EDA reports coverage rather than assuming completeness.

Individual target regressors can produce mutually inconsistent peak/trajectory
predictions. Full coherent trajectories, prediction intervals, personal calibration,
and automated feature-count selection remain future work. SHAP explains model
behavior, not causation. Synthetic data is still a separate challenge requirement.
Only load trusted joblib artifacts, since that format can execute code.
