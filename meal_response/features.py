"""Causal seven-day features and consistent curve-derived endpoints."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .data import NUTRIENTS, Patient

MINUTES = np.arange(5, 121, 5)
TARGETS = [f"glucose_{minute}" for minute in MINUTES]
META = ["participant", "source", "cohort", "time", "meal_id", "baseline", "history_days", "supplied_label_fraction", "macro_confidence"]


def summarize(curves, baseline):
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    if curves.shape[1] != 24 or len(baseline) != len(curves):
        raise ValueError("Expected 24 curve values and one baseline per meal")
    delta = np.column_stack([np.zeros(len(curves)), curves - baseline[:, None]])
    # Explicit trapezoids work with both NumPy 1.x and 2.x.
    area = lambda x: 5 * ((x[:, :-1] + x[:, 1:]) / 2).sum(axis=1)
    return pd.DataFrame({**{f"glucose_{m}": curves[:, m // 5 - 1] for m in [30, 60, 90, 120]},
                         "peak": curves.max(axis=1), "peak_rise": curves.max(axis=1) - baseline,
                         "time_to_peak": MINUTES[curves.argmax(axis=1)],
                         "iauc": area(np.maximum(delta, 0)), "net_iauc": area(delta)})


def interpolate_history(glucose, grid, max_gap):
    """Interpolate only between already-available measurements; hold at most 5 min."""
    if glucose.empty:
        return np.full(len(grid), np.nan)
    observed = glucose.time.astype("int64").to_numpy()
    values = glucose.glucose.to_numpy()
    query = pd.DatetimeIndex(grid).asi8
    right = np.searchsorted(observed, query, side="left")
    output = np.full(len(query), np.nan)
    for j, (q, r) in enumerate(zip(query, right)):
        if r < len(observed) and observed[r] == q:
            output[j] = values[r]
        elif r > 0 and r < len(observed) and observed[r] - observed[r - 1] <= max_gap * 60e9:
            fraction = (q - observed[r - 1]) / (observed[r] - observed[r - 1])
            output[j] = values[r - 1] + fraction * (values[r] - values[r - 1])
        elif r == len(observed) and q - observed[-1] <= 5 * 60e9:
            output[j] = values[-1]
    return output


def clean_meals(patient):
    meals = patient.meals.sort_values("time").copy()
    duplicate = meals.time.duplicated(keep=False)
    if duplicate.any():
        for _, group in meals.loc[duplicate].groupby("time"):
            if group[NUTRIENTS].nunique(dropna=False).max() > 1:
                raise ValueError(f"Conflicting meals at the same timestamp: {patient.participant}")
    return meals.drop_duplicates("time").reset_index(drop=True)


def make_features(patient, time, meal, enhanced=False):
    """Same function for offline training and online inference; never reads labels."""
    time = pd.Timestamp(time)
    start = time - pd.Timedelta(days=7)
    supplied = patient.glucose
    if patient.source == "t2dm":
        supplied = supplied.loc[~supplied.interpolated]
        max_gap, cadence = 15, 15
        cutoff = time
    elif patient.source == "cg_macros":
        # These local exports interpolate the original five-minute Dexcom values
        # onto minute rows without identifying originals. A 5-minute delay keeps
        # plausible interpolation right endpoints before the prediction time.
        cutoff = time - pd.Timedelta(minutes=5)
        max_gap, cadence = 10, 1
    else:
        raise ValueError(f"Unknown source: {patient.source}")
    past = supplied.loc[supplied.time.between(start, cutoff)].sort_values("time")
    if past.empty or time - past.time.iloc[-1] > pd.Timedelta(minutes=5):
        raise ValueError("stale_baseline")
    baseline = float(past.glucose.iloc[-1])
    grid = pd.date_range(time - pd.Timedelta(minutes=60), time, freq="5min")
    recent = interpolate_history(past, grid, max_gap)
    if np.isfinite(recent).sum() < 10:
        raise ValueError("insufficient_recent_cgm")
    features = {}
    for name in NUTRIENTS:
        value = meal.get(name, np.nan)
        try:
            features[name] = np.nan if pd.isna(value) else float(value)
        except (TypeError, ValueError):
            raise ValueError("invalid_nutrition") from None
        if name != "fiber" and (not np.isfinite(features[name]) or features[name] < 0):
            raise ValueError("invalid_nutrition")
    if not pd.isna(features["fiber"]) and (not np.isfinite(features["fiber"]) or features["fiber"] < 0):
        raise ValueError("invalid_nutrition")
    features["fiber_missing"] = float(pd.isna(features["fiber"]))
    features["baseline"] = baseline
    features.update({f"lag_{60 - i * 5}": float(value) for i, value in enumerate(recent)})
    valid = np.isfinite(recent)
    features.update(recent_slope=float(np.polyfit(np.arange(13)[valid] * 5, recent[valid], 1)[0]),
                    recent_std=float(np.nanstd(recent)), recent_min=float(np.nanmin(recent)),
                    recent_max=float(np.nanmax(recent)), recent_coverage=float(valid.mean()))
    hour = time.hour + time.minute / 60
    features.update(hour_sin=float(np.sin(2 * np.pi * hour / 24)), hour_cos=float(np.cos(2 * np.pi * hour / 24)))
    history_days = min(7.0, max(0.0, (time - past.time.iloc[0]).total_seconds() / 86400))
    features["history_days"] = history_days
    historical_meals = patient.meals.loc[patient.meals.time.between(start, time, inclusive="left")]
    for days in [1, 3, 7]:
        lower = time - pd.Timedelta(days=days)
        block = past.loc[past.time >= lower]
        prefix = f"history_{days}d"
        features.update({f"{prefix}_mean": float(block.glucose.mean()),
                         f"{prefix}_std": float(block.glucose.std(ddof=0)),
                         f"{prefix}_min": float(block.glucose.min()), f"{prefix}_max": float(block.glucose.max()),
                         f"{prefix}_coverage": min(1.0, len(block) / (days * 1440 / cadence + 1))})
        logged = historical_meals.loc[historical_meals.time >= lower]
        features[f"{prefix}_logged_meals"] = len(logged)
        complete = logged.loc[logged.nutrition_complete]
        features[f"{prefix}_nutrition_coverage"] = len(complete) / len(logged) if len(logged) else np.nan
        for name in NUTRIENTS:
            features[f"{prefix}_logged_{name}"] = float(complete[name].sum(min_count=1))
    if enhanced:
        if patient.source == "cg_macros":
            activity = patient.activity.loc[patient.activity.time.between(start, time)]
            for minutes in [60, 1440, 10080]:
                block = activity.loc[activity.time >= time - pd.Timedelta(minutes=minutes)]
                for column in ["mets", "hr", "activity_calories"]:
                    features[f"activity_{minutes}m_{column}_mean"] = float(block[column].mean())
                    features[f"activity_{minutes}m_{column}_coverage"] = min(1.0, block[column].notna().sum() / (minutes + 1))
        else:
            medication = patient.medication.loc[patient.medication.time.between(start, time)]
            features["medication_history_unknown"] = float(medication.empty)
            features["medication_unparsed_records"] = int((~medication.parsed.astype(bool)).sum())
            for minutes in [1440, 10080]:
                block = medication.loc[medication.time >= time - pd.Timedelta(minutes=minutes)]
                features[f"medication_{minutes}m_logged_events"] = len(block)
                for (drug, route, unit), events in block.loc[block.parsed.astype(bool)].groupby(["drug", "route", "unit"]):
                    # Sums are of recorded administrations/rates, not inferred exposure.
                    key = f"med::{minutes}::{drug}::{route}::{unit}"
                    features[key + "::recorded_sum"] = float(events.dose.sum())
                    features[key + "::count"] = len(events)
    return features


def label_curve(patient, time, observed_only=False, recording=None):
    glucose = patient.glucose
    if recording is not None and "recording" in glucose:
        glucose = glucose.loc[glucose.recording.eq(str(recording))]
    if observed_only:
        if patient.source == "cg_macros":
            raise ValueError("unknown_observation_provenance")
        glucose = glucose.loc[~glucose.interpolated]
    glucose = glucose.loc[glucose.time.between(time, time + pd.Timedelta(minutes=122.5), inclusive="right")]
    if glucose.empty:
        raise ValueError("incomplete_curve")
    times = glucose.time.astype("int64").to_numpy()
    queries = pd.DatetimeIndex([time + pd.Timedelta(minutes=int(m)) for m in MINUTES]).asi8
    right = np.searchsorted(times, queries)
    right = np.minimum(right, len(times) - 1)
    left = np.maximum(right - 1, 0)
    index = np.where(np.abs(times[left] - queries) <= np.abs(times[right] - queries), left, right)
    if (np.abs(times[index] - queries) > 2.5 * 60e9).any() or len(set(index)) != 24:
        raise ValueError("incomplete_curve")
    curve = glucose.glucose.to_numpy()[index]
    fraction = float(glucose.interpolated.to_numpy()[index].mean()) if patient.source == "t2dm" else np.nan
    return curve, fraction


def build_examples(patients, enhanced=False, observed_only=False):
    rows, audits = [], []
    for patient in patients:
        patient.meals = clean_meals(patient)
        counts = Counter()
        for _, meal in patient.meals.iterrows():
            time = meal.time
            counts["candidate_meals"] += 1
            if ((patient.meals.time > time) & (patient.meals.time <= time + pd.Timedelta(minutes=120))).any():
                counts["overlapping_meal"] += 1
                continue
            if not meal.nutrition_complete:
                counts["incomplete_nutrition"] += 1
                continue
            try:
                features = make_features(patient, time, meal, enhanced)
                curve, fraction = label_curve(patient, time, observed_only, meal.get("recording"))
            except ValueError as error:
                counts[str(error)] += 1
                continue
            rows.append({**features, "participant": patient.participant, "source": patient.source,
                         "cohort": patient.cohort, "time": time, "meal_id": meal.meal_id,
                         "macro_confidence": meal.get("macro_confidence", np.nan),
                         "supplied_label_fraction": fraction, **dict(zip(TARGETS, curve))})
            counts["usable_meals"] += 1
        audits.append({"participant": patient.participant, "source": patient.source, "cohort": patient.cohort,
                       "glucose_rows": len(patient.glucose), "flagged_interpolated_rows": int(patient.glucose.interpolated.sum()),
                       "medication_records": len(patient.medication),
                       "meals_amount_consumed_below_100": int((patient.meals.get("amount_consumed", pd.Series(dtype=float)) < 100).sum()),
                       **dict(counts)})
    return pd.DataFrame(rows), audits


def forecast(bundle, patient, time, planned_meal):
    """Return JSON-ready outcomes using the exact training feature constructor."""
    if bundle["dataset"] != "combined" and patient.source != bundle["dataset"]:
        raise ValueError("Inference source does not match the model's training dataset")
    features = make_features(patient, time, planned_meal, bundle["features_mode"] == "enhanced")
    row = pd.DataFrame([{name: features.get(name, 0.0 if name.startswith("med::") else np.nan)
                         for name in bundle["feature_names"]}])
    transformed = bundle["preprocessor"].transform(row)
    models = bundle["models"]
    curve = models.predict(transformed) if bundle["family"] == "linear" else np.column_stack([m.predict(transformed) for m in models])
    return {"minutes": MINUTES.tolist(), "glucose_mg_dl": curve[0].tolist(),
            "baseline_mg_dl": features["baseline"],
            "outcomes": summarize(curve, [features["baseline"]]).iloc[0].to_dict(),
            "assumption": "No additional meal in the next 120 minutes",
            "history_days": features["history_days"]}
