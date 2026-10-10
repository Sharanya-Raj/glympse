# CGMacros backend

Run commands from the repository root with Python 3.11 or newer:

```powershell
python -m pip install -r backend/requirements.txt
python -m backend.eda --data backend/data
python -m backend.model --data backend/data
python -m backend.explain
```

Download and extract the real participant data from
https://physionet.org/content/cgmacros/1.0.0/ into `backend/data/`.
The loader searches recursively for `CGMacros-*.csv`. Data and generated results
are excluded from Git. No real data has been downloaded or training performed yet.

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
