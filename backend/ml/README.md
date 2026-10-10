# Personalized glucose model

## Run the account/upload/predict service

From the project root:

```powershell
python -m pip install -r backend/requirements.txt
python -m backend.ml.train
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to exercise the API:

1. `POST /accounts`: username and password (minimum 12 characters). Returns a
   bearer token; use Authorize with that token. `POST /sessions` signs in again.
2. `POST /history`: upload a single-user CSV. `Date` and `CGM (mg / dl)` are
   required; dietary/treatment columns match the source data. Missing food means
   no intake under the project assumption. Explicit unavailable food stays unknown.
3. `GET /history`: check coverage and the recording's day-8 interval.
4. `POST /predict`: provide meal time, planned dietary intake, and recent CGM.
5. `GET /explanations/daily`: retrieve each forecast day's top three features.

Example prediction body (adjust dates to the returned day-8 start):

```json
{
  "meal_time": "2025-01-08T08:00:00Z",
  "dietary_intake": "Rice 100 g Egg 50 g",
  "recent_cgm": [
    {"timestamp": "2025-01-08T07:30:00Z", "glucose": 110},
    {"timestamp": "2025-01-08T07:40:00Z", "glucose": 112},
    {"timestamp": "2025-01-08T07:50:00Z", "glucose": 114},
    {"timestamp": "2025-01-08T08:00:00Z", "glucose": 115}
  ]
}
```

Dates are converted to UTC; timestamps without offsets are interpreted as UTC.
Days are 24-hour blocks from the uploaded recording's first timestamp. Need seven
preceding days, at least 230 valid CGM readings per day and duration through the
last 5-minute slot. Recent inputs need at least four distinct readings within
30 minutes of the meal, latest within five minutes. Upload limits: 10 MB / 100,000
rows. Multiple source patients or recordings are rejected. Account identity owns
all uploads; supplied Patient IDs do not grant access to anyone else's records.
Incremental uploads merge by timestamp (new values replace matching timestamps).
Changes to history invalidate saved predictions/rankings to prevent stale results.

## What runs when

`train.py` creates one versioned, trusted `backend/artifacts/population.joblib`
from other patients' completed first-seven-day meal examples. Training is a
separate operator action; signup/upload never retrains the shared model.

For each user's day-8 prediction, `service.py` derives completed meal responses
from their days 1–7, then fits regularized residual weights and a shrunken offset.
The shared imputer, range clipping and standardization are reused, not fitted on
the new user's data. Fewer than five usable meals falls back to population-only.
The personal weights are deterministically rebuilt from the seven-day window
for each request, rather than persisting a potentially stale user pickle.

Day 9 uses days 2–8 after more history is uploaded; day 10 uses days 3–9. Uploaded
outcomes from the forecast day are excluded from personal training. Features at
the requested meal use only data through its timestamp. This is conditional on
known meal/pre-meal context, not a whole-day forecast from seven days alone.

## Daily top-three features

At each prediction, compute exact feature contributions using combined shared
and personal linear weights and that meal's transformed values. Include the direct
observed-baseline component in the baseline feature, relative to population mean
baseline. Intercepts and personal offset are separate from the feature ranking.

For each user/recording/forecast day, average **absolute** contribution across all
saved meals and all four horizons, then keep the largest three. Also report mean
signed contribution (`higher`, `lower`, or `balanced`) and prediction count.
Ranking uses magnitude so positive and negative effects do not cancel. A large
importance with small mean signed effect indicates mixed directions.
No real outcome is needed to explain a forecast. Results reflect only meals
predicted so far that day, not a complete day until all meals are predicted.
Retries replace the same meal forecast, preventing double counting. Rankings
only combine the current shared-model version; history uploads clear prior results.
If fewer than three features contribute, fewer than three are returned.

The endpoint updates as predictions are made; it is not a scheduled notification.
Feature names are actual prepared inputs (including quality flags), not necessarily
three food categories. Keyword food groups overlap and grams are food weight,
not carbohydrates. Treatment logs are not pharmacokinetic exposure estimates.
Contributions explain model behavior, not causality or treatment recommendations.

## Tests and offline benchmarks

```powershell
python -m unittest backend.ml.test_model backend.test_api -v
python -m backend.ml.evaluate
python -m backend.ml.benchmarks
```

Evaluation holds each patient out of population training and uses rolling personal
history. Metrics include MAE/RMSE, R2/change-R2, MAPE and accuracy within stated
tolerances. `100-MAPE` is not accuracy. This development dataset has already informed
implementation; final performance needs an untouched external cohort.
The offline evaluator saves `daily_top_features.csv` with per-patient/day rankings.

Both evaluation and benchmark commands regenerate these files in
`backend/artifacts/personalized-meal-regression/`:

- `day8_actual_vs_predicted.csv`: actual and predicted day-8 glucose for each
  patient, recording, meal and horizon, with signed/absolute errors and tolerance flags.
- `day8_accuracy.csv`: day-8 scores per patient and recording.
- `daily_accuracy.csv`: the same scores for every evaluated day, including later
  rolling windows. Each day has a combined `all` horizon row and separate
  30/60/90/120-minute rows.

Daily scores include MAE/RMSE in mg/dL, MAPE, R2, and the percentage of forecasts
within 20 mg/dL or 15% of actual glucose. Accuracy means these stated tolerances;
there is no single classification accuracy for regression. R2 is blank with fewer
than two forecasts. The `forecasts` and `meals` counts show each score's sample size.
These compare post-meal forecasts, not a continuous whole-day glucose trace.

## Files and deployment limits

- `model.py`: shared feature extraction, Ridge fit, personal adapter, prediction.
- `service.py`: validate raw uploads and make outcome-free forecasts.
- `train.py`: create the shared deployment artifact.
- `prepare.py`, `normalize.py`: raw food/treatment preparation.
- `explain.py`: exact contributions and daily ranking.
- `evaluate.py`, `benchmarks.py`: offline evaluation, separate from request handling.
- `test_model.py` and `backend/test_api.py`: correctness and account-flow tests.

Local accounts use salted scrypt passwords, hashed bearer sessions (24 hours),
and per-user histories/forecasts in `backend/runtime/accounts.sqlite3`. Configure
GLYIMPSE_STATE_DIR and GLYIMPSE_MODEL_PATH as needed. This is a local backend:
frontend account screens are not wired, and the existing Lambda ZIP is stale.
Production hosting needs persistent private storage, HTTPS and account abuse
controls; Lambda /tmp is not a durable user database. Raw local SQLite data is
not encrypted by this application. No service has been deployed by these commands.
