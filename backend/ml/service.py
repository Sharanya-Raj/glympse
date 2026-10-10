"""Raw personal CSV validation, day-8 adaptation, and outcome-free prediction."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .model import (CONTEXT, HORIZONS, fit_personal, input_features,
                    meal_examples, personal_history, predict)
from .normalize import enrich
from .prepare import prepare
from .explain import contributions

COLUMNS = ["Recording", "Patient", "Date", "CGM (mg / dl)", "CBG (mg / dl)",
           "Dietary intake", "Insulin dose - s.c.", "Non-insulin hypoglycemic agents",
           "CSII - bolus insulin (Novolin R, IU)", "CSII - basal insulin (Novolin R, IU / H)"]


def validate_history(data):
    if data.empty or len(data) > 100000:
        raise ValueError("Upload between 1 and 100,000 rows")
    if not {"Date", "CGM (mg / dl)"}.issubset(data.columns):
        raise ValueError("CSV needs Date and CGM (mg / dl) columns")
    for col in ["Patient", "Recording"]:
        if col in data and data[col].dropna().nunique() > 1:
            raise ValueError("Upload one patient and one recording per account")
    data = data.copy()
    for col in COLUMNS:
        if col not in data:
            data[col] = pd.NA
    data["Date"] = pd.to_datetime(data.Date, errors="raise", utc=True).dt.tz_convert(None)
    if data.Date.isna().any() or data.Date.duplicated().any():
        raise ValueError("Dates must be valid and unique")
    glucose = pd.to_numeric(data["CGM (mg / dl)"], errors="raise")
    if (glucose.notna() & (~np.isfinite(glucose) | (glucose <= 0))).any():
        raise ValueError("CGM values must be positive finite numbers or blank")
    data["CGM (mg / dl)"] = glucose
    # Ownership comes from authentication, never identifiers supplied in the CSV.
    data["Patient"], data["Recording"] = "current_user", "active"
    return data[COLUMNS].sort_values("Date").reset_index(drop=True)


def prepared_history(raw):
    raw = raw.copy()
    for col in COLUMNS:
        if col != "Date":
            raw[col] = raw[col].astype("string")
    return enrich(prepare(raw))


def readiness(raw):
    anchor = raw.Date.min()
    cutoff = anchor + pd.Timedelta(days=7)
    first_week = raw[raw.Date < cutoff]
    blocks = ((first_week.Date - anchor).dt.total_seconds() // 86400).astype(int)
    counts = first_week.groupby(blocks)["CGM (mg / dl)"].count().reindex(range(7), fill_value=0)
    ready = bool(raw.Date.max() >= cutoff - pd.Timedelta(minutes=5) and (counts >= 230).all())
    return {"ready_for_day8": ready, "day8_start": cutoff.isoformat(),
            "day8_end": (cutoff + pd.Timedelta(days=1)).isoformat(),
            "daily_cgm_counts": counts.astype(int).tolist()}


def forecast_meal(raw, meal_time, dietary_intake, recent_cgm, bundle):
    timestamp = pd.to_datetime(meal_time, utc=True).tz_convert(None)
    day = int((timestamp - raw.Date.min()).total_seconds() // 86400) + 1
    if day < 8:
        raise ValueError("Personal forecasts start on day 8 after seven days of history")
    cutoff = raw.Date.min() + pd.Timedelta(days=day - 1)
    window_start = cutoff - pd.Timedelta(days=7)
    history_raw = raw[(raw.Date >= window_start) & (raw.Date < cutoff)].copy()
    blocks = ((history_raw.Date - window_start).dt.total_seconds() // 86400).astype(int)
    counts = history_raw.groupby(blocks)["CGM (mg / dl)"].count().reindex(range(7), fill_value=0)
    if history_raw.empty or history_raw.Date.max() < cutoff - pd.Timedelta(minutes=5) or (counts < 230).any():
        raise ValueError("Need seven preceding days with at least 230 CGM readings each")
    # Ignore uploaded outcomes from the forecast day when fitting personal weights.
    completed = prepared_history(history_raw)
    try:
        examples, _, _ = meal_examples(completed)
        history = personal_history(examples, "active", cutoff)
    except ValueError as exc:
        if str(exc) != "No usable meal-response examples":
            raise
        history = pd.DataFrame()
    features, model = bundle["features"], bundle["model"]
    coef, intercept = fit_personal(model, history, features)
    context_rows = []
    for reading in recent_cgm:
        time = pd.to_datetime(reading["timestamp"], utc=True).tz_convert(None)
        if not timestamp - pd.Timedelta(minutes=30) <= time <= timestamp:
            raise ValueError("Recent CGM timestamps must be in the 30 minutes at/before the meal")
        context_rows.append({"Date": time, "CGM (mg / dl)": reading["glucose"]})
    context = pd.DataFrame(context_rows)
    if context.empty or context.Date.duplicated().any():
        raise ValueError("Provide distinct recent CGM timestamps")
    # Uploaded context through this meal is permitted; future rows are never used.
    combined = raw[raw.Date <= timestamp].copy().set_index("Date")
    context = context.set_index("Date")
    combined = combined.reindex(combined.index.union(context.index))
    combined.loc[context.index, "CGM (mg / dl)"] = context["CGM (mg / dl)"]
    combined = combined.reset_index().rename(columns={"index": "Date"}).sort_values("Date")
    if timestamp not in set(combined.Date):
        combined = pd.concat([combined, pd.DataFrame([{"Date": timestamp}])], ignore_index=True)
    combined.loc[combined.Date == timestamp, "Dietary intake"] = dietary_intake
    combined["Patient"], combined["Recording"] = "current_user", "active"
    combined = validate_history(combined)
    current = prepared_history(combined).set_index("Date").sort_index()
    factors = [col for col in features if col not in CONTEXT]
    for col in factors:
        current[col] = pd.to_numeric(current[col], errors="coerce").astype(float)
    current["CGM (mg / dl)"] = pd.to_numeric(current["CGM (mg / dl)"], errors="coerce").astype(float)
    x = pd.DataFrame([input_features(current, timestamp, factors)])[features]
    delta = predict(model, coef, intercept, x)[0]
    baseline = float(x.baseline.iloc[0])
    effects, reference = contributions(model, coef, features, x)
    predictions = []
    all_effects = []
    for j, horizon in enumerate(HORIZONS):
        effect = effects[0, j]
        top = np.argsort(np.abs(effect))[-3:][::-1]
        all_effects.extend({"feature": feature, "horizon_minutes": horizon,
                           "contribution": float(effect[k])} for k, feature in enumerate(features))
        predictions.append({"minutes_after_meal": horizon, "glucose_mg_dl": float(baseline + delta[j]),
            "change_mg_dl": float(delta[j]), "top_features": [
                {"feature": features[k], "contribution_mg_dl": float(effect[k])} for k in top if abs(effect[k]) > 1e-10]})
    return {"model_version": bundle["version"], "meal_time": timestamp.isoformat(),
            "baseline_glucose": baseline, "personal_training_meals": len(history),
            "personalized": len(history) >= 5, "personal_cutoff": cutoff.isoformat(),
            "day": day, "baseline_reference": reference,
            "predictions": predictions, "feature_contributions": all_effects}


def load_population(path):
    bundle = joblib.load(Path(path))  # Trusted server-owned artifact only.
    if bundle.get("schema") != 1:
        raise ValueError("Unsupported model schema")
    return bundle
