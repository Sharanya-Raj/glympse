"""Leave-one-patient-out meal regression with rolling seven-day personal adaptation."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
HORIZONS = [30, 60, 90, 120]
TARGETS = [f"delta_{h}" for h in HORIZONS]
CONTEXT = ["baseline", "premeal_slope", "premeal_mean", "premeal_std", "hour_sin", "hour_cos"]


def input_features(frame, timestamp, factors):
    """Build one meal input without requiring or reading its future outcome."""
    past = frame.loc[:timestamp]
    glucose = past["CGM (mg / dl)"].dropna()
    before = glucose.loc[timestamp - pd.Timedelta(minutes=30):timestamp]
    if len(before) < 4 or (timestamp - before.index[-1]).total_seconds() > 300:
        raise ValueError("Need at least four pre-meal CGM readings in 30 minutes; latest within five minutes")
    elapsed = (before.index[-1] - before.index[0]).total_seconds() / 60
    if elapsed <= 0:
        raise ValueError("Pre-meal readings need distinct timestamps")
    angle = 2 * np.pi * (timestamp.hour + timestamp.minute / 60) / 24
    event = past.loc[timestamp]
    row = {"baseline": float(before.iloc[-1]),
           "premeal_slope": (before.iloc[-1] - before.iloc[0]) / elapsed,
           "premeal_mean": before.mean(), "premeal_std": before.std(),
           "hour_sin": np.sin(angle), "hour_cos": np.cos(angle)}
    for col in factors:
        if col.startswith("food_"):
            row[col] = event[col]
        else:
            history = past.loc[timestamp - pd.Timedelta(hours=24):timestamp, col]
            row[col] = history.mean() if "basal" in col else history.sum(min_count=1)
    return row


def meal_examples(data):
    data = data.copy()
    data["Date"] = pd.to_datetime(data.Date, errors="raise")
    factors = [c for c in data if c.startswith(("food_", "insulin_", "med_", "csii_"))
               and not c.endswith("normalized") and c != "food_status"]
    for col in ["CGM (mg / dl)", *factors]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    rows, coverage = [], []
    for (patient, recording), frame in data.groupby(["Patient", "Recording"]):
        frame = frame.sort_values("Date").set_index("Date")
        if frame.index.duplicated().any():
            raise ValueError(f"Duplicate timestamps in {recording}")
        start, end = frame.index.min(), frame.index.max()
        food = frame[frame.food_logged_now.eq(1)]
        glucose = frame["CGM (mg / dl)"].dropna()
        valid = 0
        for timestamp, event in food.iterrows():
            label_end = timestamp + pd.Timedelta(minutes=120)
            if label_end > end or ((food.index > timestamp) & (food.index <= label_end)).any():
                continue
            before = glucose.loc[timestamp - pd.Timedelta(minutes=30):timestamp]
            if len(before) < 4 or (timestamp - before.index[-1]).total_seconds() > 300:
                continue
            targets = []
            for horizon in HORIZONS:
                target_time = timestamp + pd.Timedelta(minutes=horizon)
                left, right = glucose.loc[:target_time], glucose.loc[target_time:]
                if left.empty or right.empty:
                    break
                a, b = left.index[-1], right.index[0]
                gap = (b - a).total_seconds()
                if gap > 900:
                    break
                fraction = (target_time - a).total_seconds() / gap if gap else 0
                targets.append(float(left.iloc[-1] + fraction * (right.iloc[0] - left.iloc[-1])))
            if len(targets) != 4:
                continue
            context = input_features(frame, timestamp, factors)
            baseline = context["baseline"]
            row = {"Patient": str(patient), "Recording": str(recording), "meal_time": timestamp,
                   "label_end": label_end, "recording_start": start,
                   "day": int((timestamp - start).total_seconds() // 86400) + 1,
                   **context}
            row.update({target: glucose_value - baseline for target, glucose_value in zip(TARGETS, targets)})
            rows.append(row)
            valid += 1
        coverage.append({"Patient": patient, "Recording": recording,
                         "logged_meals": len(food), "usable_meals": valid,
                         "duration_days": (end - start).total_seconds() / 86400})
    if not rows:
        raise ValueError("No usable meal-response examples")
    return pd.DataFrame(rows), CONTEXT + factors, pd.DataFrame(coverage)


def fit_population(examples, features, excluded_patient):
    # Only other patients' first-seven-day outcomes can train the population model.
    cutoff = examples.recording_start + pd.Timedelta(days=7)
    training = examples[(examples.Patient != excluded_patient) & (examples.label_end <= cutoff)]
    if len(training) < 20:
        raise ValueError("Fewer than 20 eligible population training meals")
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          # Bound unseen feature values to population-training ranges.
                          # Linear dose extrapolation can otherwise become extreme.
                          MinMaxScaler(clip=True), StandardScaler(), Ridge(alpha=100.0))
    model.fit(training[features], training[TARGETS])
    return model, sorted(training.Patient.unique())


def fit_personal(model, history, features):
    """Fit residual coefficients with shrinkage; sparse histories use population only."""
    count = len(history)
    coefficients = np.zeros_like(model[-1].coef_)
    intercept = np.zeros(4)
    if count >= 5:
        x = model[:-1].transform(history[features])
        residuals = history[TARGETS].to_numpy() - model.predict(history[features])
        # Shrink the patient's average correction toward zero for short histories.
        intercept = residuals.mean(axis=0) * count / (count + 10)
        adapter = Ridge(alpha=100.0, fit_intercept=False)
        adapter.fit(x, residuals - intercept)
        coefficients = adapter.coef_
    return coefficients, intercept


def predict(model, personal_coefficients, personal_intercept, features):
    """Return four glucose changes; caller adds the observed pre-meal baseline."""
    z = model[:-1].transform(features)
    return model.predict(features) + z @ personal_coefficients.T + personal_intercept


def personal_history(examples, recording, cutoff):
    return examples[(examples.Recording == recording) &
                    (examples.meal_time >= cutoff - pd.Timedelta(days=7)) &
                    (examples.label_end < cutoff)]


