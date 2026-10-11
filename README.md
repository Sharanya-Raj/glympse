**Website:** [https://main.d37tns3udtrqqd.amplifyapp.com/](https://main.d37tns3udtrqqd.amplifyapp.com/)

**Demo Login:**
* **Username:** `sharanyaraj24@gmail.com`
* **Password:** `@Test123Permanent!`

Glympse predicts how a person's glucose will respond to a meal **before** they eat it.
The backend learns from the first seven days of a user's continuous glucose monitor
(CGM) readings, meals, and medication, then forecasts the change in glucose at
30, 60, 90, and 120 minutes for any meal the user describes, types, photographs, or
dictates.

It is a [FastAPI](https://fastapi.tiangolo.com/) service that also serves the
web frontend, and it runs locally with Uvicorn or serverless on AWS Lambda.

---

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [How predictions work](#how-predictions-work)
- [Food services](#food-services)
- [Training the shared model](#training-the-shared-model)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security](#security)
- [Troubleshooting](#troubleshooting)
- [Research pipeline (legacy)](#research-pipeline-legacy)

---

## Features

- **Personalized meal forecasts.** A shared population model is adapted to each user
  from their own first-week meals, so the same meal can produce different forecasts
  for different people.
- **Personal spike thresholds.** A forecast counts as a *spike* when the predicted rise
  exceeds the user's 75th-percentile post-meal rise, and as a *big spike* above the
  90th percentile, both computed from their days 1–7 instead of a fixed number.
- **Sugar adjustment.** Sugar grams from USDA data, multiplied by the user's own
  rise per gram of carbohydrate, are added on top of the model and labelled as such.
- **"Try instead" alternatives.** Smaller portions or swaps (for example, skipping the
  sweet items) re-forecast at the same time and starting glucose.
- **Live day-8 replay.** A recorded day plays back reading by reading. Confirmed meals
  add their predicted effect to the replay, and the recorded value stays available.
- **Explanations.** The top contributing features for each forecast and daily
  top-three feature rankings.
- **Food input in four ways.** Typed foods, USDA search, photo recognition on the
  server, and spoken dictation.

## Architecture

```text
Browser (frontend/)
   │  HTTPS + bearer token
   ▼
FastAPI app (backend/app.py) ── served by Uvicorn locally or by AWS Lambda via Mangum
   ├── accounts.py          SQLite: users, sessions, histories, replay clock, confirmed meals
   ├── ml/                  Forecasting: shared Ridge model + personal adapter + spike rules
   │     └── artifacts/population.joblib (trusted, versioned shared model)
   └── food/                Food input
         ├── usda.py        → USDA FoodData Central API   (nutrition per portion)
         ├── identify.py    → nateraw/food ViT on server  (photo → dish name)
         └── meal.py        → ElevenLabs speech-to-text  → Google Gemini (meal parsing, USDA match)
```

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Web framework | FastAPI, Pydantic, Uvicorn |
| Data and ML | pandas, NumPy, scikit-learn (Ridge), joblib |
| Photo recognition | PyTorch, Hugging Face Transformers (`nateraw/food`, Food-101) |
| Storage | SQLite (`GLYIMPSE_STATE_DIR`) |
| External APIs | USDA FoodData Central, ElevenLabs, Google Gemini |
| Hosting | AWS Lambda (`lambda_function.py`, Mangum) |

## Project layout

```text
backend/
├── app.py               FastAPI application: auth, history, replay, predictions, static frontend
├── accounts.py          SQLite persistence (users, sessions, uploads, replay state, forecasts)
├── seed.py              Development-only demo accounts loaded from dataset workbooks
├── lambda_function.py   AWS Lambda entry point (Mangum handler)
├── requirements.txt     Runtime and test dependencies
├── .env.example         Documented configuration template
├── ml/                  Maintained forecasting pipeline (see ml/README.md)
│   ├── train.py         Builds artifacts/population.joblib
│   ├── service.py       Per-user personalization and meal forecasts
│   ├── personal.py      Personal thresholds, carb response, sugar adjustment, replay simulation
│   ├── spike.py         Spike verdicts and food drivers
│   ├── catalog.py       Food, insulin, and tablet catalogue with nutrition estimates
│   └── test_model.py    Model tests
├── food/                USDA lookup, photo identification, and dictation (see food/README.md)
├── test_api.py          API tests
└── data.py, eda.py, features.py, model.py, explain.py   Research pipeline (legacy)
```

Generated and private folders are gitignored: `data/`, `artifacts/`, `runtime/`,
`reports/`, and `.env`.

## Getting started

### Prerequisites

- Python 3.12
- The shared model at `backend/artifacts/population.joblib` (obtain it from the team or
  [train it](#training-the-shared-model))
- For demo accounts: the dataset workbooks, for example `2000_0_20201230.xlsx`

### 1. Install

From the repository root (`glyimpse/`):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
```

Optional photo recognition (installs a CPU-only build of PyTorch, which is much smaller than the GPU build):

```powershell
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r backend\food\requirements.txt
```

### 2. Configure

```powershell
Copy-Item backend\.env.example backend\.env
```

Edit `backend/.env`. A typical local setup with two demo patients:

```ini
GLYIMPSE_ENV=dev
GLYIMPSE_SEED_RECORDINGS=2000_0_20201230=jane:choose-a-password,2001_0_20201102=justin:choose-a-password
GLYIMPSE_DATA_DIR=backend/data
GLYIMPSE_ALLOW_SIGNUP=false
GLYIMPSE_ALLOWED_ORIGINS=*
USDA_API_KEY=your-usda-key
ELEVEN_LABS_API_KEY=your-elevenlabs-key
GEMINI_API_KEY=your-gemini-key
```

### 3. Run

```powershell
.\.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

- App: <http://127.0.0.1:8000>
- Interactive API docs: <http://127.0.0.1:8000/docs>

Settings are read once at startup, so restart the server after editing `.env`.

## Configuration

Settings come from environment variables. `backend/.env` is loaded first, then the
repository-root `.env` as a fallback. Real environment variables always take
precedence.

| Variable | Default | Purpose |
|---|---|---|
| `GLYIMPSE_ENV` | *(unset)* | `dev` enables demo seeding and short admin passwords. Leave unset in production. |
| `GLYIMPSE_ADMIN_USERNAME` / `GLYIMPSE_ADMIN_PASSWORD` | *(unset)* | Admin account created or re-keyed at startup. The password needs at least 12 characters outside dev. |
| `GLYIMPSE_SEED_RECORDINGS` | *(unset)* | Dev only. Comma-separated `recording[=username][:password]` entries. |
| `GLYIMPSE_SEED_PASSWORD` | *(unset)* | Password for seed entries that don't set their own. |
| `GLYIMPSE_DATA_DIR` | `data/frontend_test` | Folder containing the seed workbooks. |
| `GLYIMPSE_ALLOW_SIGNUP` | `false` | Enables public `POST /accounts`. |
| `GLYIMPSE_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Use exact origins in production. |
| `GLYIMPSE_STATE_DIR` | `backend/runtime` | Location of the SQLite database. |
| `GLYIMPSE_MODEL_PATH` | `backend/artifacts/population.joblib` | Shared model artifact. |
| `GLYIMPSE_FRONTEND_DIR` | `frontend` | Static frontend to serve. An empty value gives an API-only server. |
| `GLYIMPSE_EVALUATION_PATH` | *(unset)* | Offline day-8 results for `GET /model/accuracy`. |
| `USDA_API_KEY` (or `FDC_API_KEY`) | *(unset)* | USDA FoodData Central key. |
| `ELEVEN_LABS_API_KEY` | *(unset)* | ElevenLabs speech-to-text, for dictation. |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`, `GOOGLE_API_AI`) | *(unset)* | Google Gemini, for dictation. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model name. |
| `FOOD_MODEL_NAME` | `nateraw/food` | Hugging Face image classifier. |
| `FOOD_CONFIDENCE_THRESHOLD` | `0.6` | Below this, the UI asks the user to confirm the photo guess. |

Seeding renames an existing `patient<id>` account to the configured username, so its
history, replay position, and meals carry over. Existing histories are never overwritten.

## API reference

Every endpoint except `/health`, `/catalog`, `POST /accounts`, and `POST /sessions`
requires `Authorization: Bearer <token>`. Full request and response schemas are
listed at `/docs`.

### Accounts and sessions

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status, model availability, demo usernames |
| `POST` | `/accounts` | Create an account (only when sign-up is enabled) |
| `POST` | `/sessions` | Sign in and receive `access_token` |
| `DELETE` | `/sessions` | Sign out |
| `GET` | `/me` | Current username and admin flag |

### History and replay

| Method | Path | Description |
|---|---|---|
| `POST` | `/history` | Upload a single-patient CSV (at most 10 MB and 100,000 rows) |
| `GET` | `/history` | Coverage, readiness for day 8, and personal thresholds |
| `GET` | `/history/readings` | All readings and events up to the replay clock |
| `GET` | `/live?since=` | Replay state, optionally only rows after `since` |
| `POST` | `/live/control` | `play`, `pause`, `seek`, `speed`, or `restart` |

### Predictions

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Forecast a meal. `confirm: true` saves it to the replay. |
| `POST` | `/predict/alternatives` | Ranked "try instead" meals with their own verdicts |
| `GET` | `/meals` | Confirmed meals with forecast, meal effect, and sugar adjustment |
| `GET` | `/explanations/daily` | Top-three features per forecast day |
| `GET` | `/model/accuracy` | Offline day-8 accuracy (needs `GLYIMPSE_EVALUATION_PATH`) |
| `GET` | `/catalog` | Foods, portions, insulins, and tablets for the meal builder |

Example request:

```json
POST /predict
{
  "dietary_intake": "Rice 100 g, Cake 80 g",
  "foods": [
    {"name": "Rice", "grams": 100, "carbs_g": 28, "sugar_g": 0.1},
    {"name": "Cake", "grams": 80, "carbs_g": 44, "sugar_g": 28}
  ],
  "insulin": {"name": "Novolin R", "dose": 6}
}
```

When `meal_time` is omitted, the meal is placed at the replay clock's current time.

### Food

| Method | Path | Description |
|---|---|---|
| `GET` | `/food/search?q=` | Search USDA foods |
| `GET` | `/food/nutrients?label=` | Nutrients and portions for a dish label |
| `GET` | `/food/nutrients/{fdc_id}` | Nutrients and portions for a USDA food |
| `POST` | `/food/identify` | Photo (JPG, PNG, or WebP, at most 10 MB) to the top five dish guesses |
| `POST` | `/food/dictate` | Audio to transcript, parsed items, and USDA matches |

## How predictions work

1. **Shared model.** `ml/train.py` fits a regularized (Ridge) model on completed meal
   responses from other patients' first seven days and saves a versioned artifact.
2. **Personalization.** For each request, `ml/service.py` rebuilds the user's meal
   responses from days 1–7 and fits shrunken residual weights. Users with fewer than
   five usable meals fall back to the population model.
3. **Forecast.** Features use only data available at the meal time. Outputs are the
   glucose change at +30, +60, +90, and +120 minutes, with per-feature contributions.
4. **Sugar adjustment.** The model weights sugar weakly, so `ml/personal.py` adds
   sugar grams × the user's mg/dL per gram of carbs (clipped to 0.5–5, default 2.0),
   shaped over two hours and capped at 150 mg/dL.
5. **Verdict.** `ml/spike.py` compares the predicted rise with the user's personal
   75th and 90th percentile rises.
6. **Confirmed meals.** The meal effect (forecast with the meal minus a "no food"
   forecast) is interpolated onto the replay. Simulated readings keep a `recorded` field.

See [`ml/README.md`](ml/README.md) for validation rules, data requirements, and limits.

## Food services

| Service | Used for | Key |
|---|---|---|
| USDA FoodData Central | Calories, carbs, sugar, protein, fat, and fiber per portion | `USDA_API_KEY` |
| `nateraw/food` (Hugging Face) | Photo recognition of 101 dishes, run on the server | None |
| ElevenLabs | Speech-to-text for dictated meals | `ELEVEN_LABS_API_KEY` |
| Google Gemini | Extracts foods and amounts from the transcript and picks USDA matches | `GEMINI_API_KEY` |

API keys stay on the server, and the frontend never sees them. The photo model
downloads once (about 650 MB) to the Hugging Face cache on first use; loading takes
20–40 s, and each photo after that takes about 0.2 s on a CPU. See
[`food/README.md`](food/README.md) for details.

## Training the shared model

```powershell
.\.venv\Scripts\python -m backend.ml.train --data path\to\5min_glucose.csv
```

Always pass `--data` explicitly; the built-in default points to a developer's local
path. The output is `backend/artifacts/population.joblib`. Training is an operator
action, and sign-up or upload never retrains the model.

> **Warning:** joblib files can execute code when loaded. Only use artifacts from a trusted source.

## Testing

```powershell
$env:PYTHONWARNINGS = "ignore"
.\.venv\Scripts\python -m unittest backend.ml.test_model backend.test_api -v
.\.venv\Scripts\python -m unittest discover -s backend -t .
```

Frontend unit tests:

```powershell
node --test frontend\portions.test.cjs frontend\voice.test.cjs
```

`backend/test_pipeline.py` covers the legacy research pipeline and needs `xgboost`.

## Deployment

`lambda_function.py` exposes `handler = Mangum(app)` for AWS Lambda behind an HTTPS
endpoint. Before deploying:

- **Persist state outside Lambda.** The Lambda filesystem is temporary, so the SQLite
  database in `GLYIMPSE_STATE_DIR` is not durable there. Use mounted storage (for
  example, Amazon EFS) or move persistence to a managed database.
- **Photo recognition and size limits.** PyTorch plus the model exceeds Lambda's
  250 MB zip limit. Use a container image or host photo identification separately;
  `/food/identify` returns a clean `503` when the model is unavailable.
- **Production settings.** Leave `GLYIMPSE_ENV` unset, set a strong admin password,
  list exact origins in `GLYIMPSE_ALLOWED_ORIGINS`, keep sign-up off unless needed,
  and provide API keys through the platform's secret store.
- **Ship the model.** Include `population.joblib` in the package or point
  `GLYIMPSE_MODEL_PATH` at it.

## Security

- Passwords are salted and hashed, and sessions are random bearer tokens. A password
  change ends existing sessions.
- Each account can only read its own uploads. A patient ID inside an uploaded file
  grants no access to anyone else's records.
- Uploads are size-limited and validated, and photos and audio are type-checked.
- Secrets live in environment variables, and `.env` files are gitignored. Never
  commit API keys, datasets, `runtime/`, or `artifacts/`.
- Demo seeding refuses to run outside development mode.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/food/dictate` returns 503 | A key is missing, or the server started before the key was added. Check `.env` and restart. |
| `/food/identify` returns 503 | `torch` or `transformers` isn't installed. Install `food/requirements.txt`. |
| "Shared model missing" | `artifacts/population.joblib` is absent. Obtain or train it. |
| Startup fails with "No workbook named …" | `GLYIMPSE_DATA_DIR` doesn't contain the seed `.xlsx` files. |
| Spike lines show "Default" | The server is running older code or the user has too few clean meals. Restart the server, or check `GET /history` → `personal`. |
| `ModuleNotFoundError: google` | Run `pip install -r backend/requirements.txt` again (it includes `google-genai`). |

## Research pipeline (legacy)

`data.py`, `eda.py`, `features.py`, `model.py`, and `explain.py` are an earlier XGBoost
and SHAP pipeline used by `meal_response/comparison_notebook.py`. They aren't used by
the API.

```powershell
python -m backend.eda --data backend/data       # data-quality reports → backend/reports/
python -m backend.model --data backend/data     # XGBoost multi-output model → backend/artifacts/
python -m backend.explain                       # day-8 SHAP rankings
```

Inputs are either the prepared forecasts in `data/sharanya_cg/` (`train.csv`,
`validation.csv`, `test.csv`) or raw [CGMacros](https://physionet.org/content/cgmacros/1.0.0/)
files (`CGMacros-*.csv`) under `backend/data/`. Training uses study days 1–7 and
evaluation uses day 8. SHAP values describe model behaviour, not causation.
