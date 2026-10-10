"""Leave-one-patient-out meal regression with rolling seven-day personal adaptation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
HORIZONS = [30, 60, 90, 120]
TARGETS = [f"delta_{h}" for h in HORIZONS]
CONTEXT = ["baseline", "premeal_slope", "premeal_mean", "premeal_std", "hour_sin", "hour_cos"]


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
            baseline = float(before.iloc[-1])
            minutes = (before.index[-1] - before.index[0]).total_seconds() / 60
            angle = 2 * np.pi * (timestamp.hour + timestamp.minute / 60) / 24
            row = {"Patient": str(patient), "Recording": str(recording), "meal_time": timestamp,
                   "label_end": label_end, "recording_start": start,
                   "day": int((timestamp - start).total_seconds() // 86400) + 1,
                   "baseline": baseline, "premeal_slope": (before.iloc[-1] - before.iloc[0]) / minutes,
                   "premeal_mean": before.mean(), "premeal_std": before.std(),
                   "hour_sin": np.sin(angle), "hour_cos": np.cos(angle)}
            for col in factors:
                if col.startswith("food_"):
                    row[col] = event[col]
                else:
                    # Treatment log summaries available at the meal; no future events.
                    history = frame.loc[timestamp - pd.Timedelta(hours=24):timestamp, col]
                    row[col] = history.mean() if "basal" in col else history.sum(min_count=1)
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


def run(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples, features, coverage = meal_examples(data)
    coverage.to_csv(output / "meal_eligibility.csv", index=False)
    predictions, explanations, training_audit, artifacts = [], [], [], {}
    for patient, personal in examples.groupby("Patient"):
        test = personal[personal.day >= 8]
        if test.empty:
            continue
        model, population_ids = fit_population(examples, features, patient)
        assert patient not in population_ids
        training_audit.append({"Patient": patient, "population_patients": "|".join(population_ids)})
        for (recording, day), upcoming in test.groupby(["Recording", "day"]):
            start = upcoming.recording_start.iloc[0] + pd.Timedelta(days=int(day) - 1)
            # Refit once per forecast day, using completed outcomes from its previous 7 days.
            history = personal_history(personal, recording, start)
            coef, intercept = fit_personal(model, history, features)
            z = model[:-1].transform(upcoming[features])
            pooled = model.predict(upcoming[features])
            adapted = predict(model, coef, intercept, upcoming[features])
            combined = model[-1].coef_ + coef
            for i, (_, meal) in enumerate(upcoming.iterrows()):
                for j, horizon in enumerate(HORIZONS):
                    actual = meal.baseline + meal[TARGETS[j]]
                    predictions.append({"Patient": patient, "Recording": recording, "day": day,
                        "meal_time": meal.meal_time, "horizon_minutes": horizon,
                        "personal_training_meals": len(history), "population_only": len(history) < 5,
                        "personal_history_start": start - pd.Timedelta(days=7),
                        "personal_cutoff": start, "actual_glucose": actual,
                        "predicted_glucose": meal.baseline + adapted[i, j],
                        "population_prediction": meal.baseline + pooled[i, j],
                        "baseline_prediction": meal.baseline})
                    # Exact additive contributions in standardized model space.
                    for k, feature in enumerate(features):
                        explanations.append({"Patient": patient, "Recording": recording, "day": day,
                            "meal_time": meal.meal_time, "horizon_minutes": horizon, "feature": feature,
                            "contribution": z[i, k] * combined[j, k]})
                    reconstructed = (meal.baseline + model[-1].intercept_[j] + intercept[j]
                                     + np.sum(z[i] * combined[j]))
                    assert np.isclose(reconstructed, meal.baseline + adapted[i, j])
            artifacts[(patient, recording, int(day))] = {
                "population": model, "personal_coefficients": coef,
                "personal_intercept": intercept, "features": features,
                "horizons": HORIZONS, "cutoff": str(start), "training_meals": len(history)}
    if not predictions:
        raise ValueError("No usable meals on day 8 or later")
    results = pd.DataFrame(predictions)
    results["absolute_error"] = (results.actual_glucose - results.predicted_glucose).abs()
    results.to_csv(output / "meal_predictions.csv", index=False)
    results[results.day == 8].to_csv(output / "day8_predictions.csv", index=False)
    scores = []
    for (patient, horizon), group in results.groupby(["Patient", "horizon_minutes"]):
        scores.append({"Patient": patient, "horizon_minutes": horizon, "meals": len(group),
            "r2": r2_score(group.actual_glucose, group.predicted_glucose) if len(group) >= 2 else np.nan,
            "mae": mean_absolute_error(group.actual_glucose, group.predicted_glucose),
            "rmse": np.sqrt(mean_squared_error(group.actual_glucose, group.predicted_glucose)),
            "population_mae": mean_absolute_error(group.actual_glucose, group.population_prediction),
            "baseline_mae": mean_absolute_error(group.actual_glucose, group.baseline_prediction)})
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "patient_metrics.csv", index=False)
    effects = pd.DataFrame(explanations)
    effects.to_csv(output / "feature_contributions.csv", index=False)
    ranking = effects.assign(absolute_contribution=effects.contribution.abs()).groupby(
        ["Patient", "horizon_minutes", "feature"]).agg(
            mean_absolute_contribution=("absolute_contribution", "mean"),
            mean_signed_contribution=("contribution", "mean")).reset_index()
    ranking.sort_values(["Patient", "horizon_minutes", "mean_absolute_contribution"],
                        ascending=[True, True, False]).to_csv(output / "patient_feature_importance.csv", index=False)
    pd.DataFrame(training_audit).to_csv(output / "population_training_audit.csv", index=False)
    joblib.dump(artifacts, output / "personalized_models.joblib")
    summary = {"patients_evaluated": int(results.Patient.nunique()),
               "meal_forecasts": int(len(results) // 4),
               "day8_meal_forecasts": int((results.day == 8).sum() // 4),
               "protocol": "Leave-one-patient-out population; rolling prior-7-day residual adaptation; fixed alpha=100",
               "horizons": {str(h): {"personalized_mae": float(g.absolute_error.mean()),
                    "population_mae": float((g.actual_glucose - g.population_prediction).abs().mean()),
                    "baseline_mae": float((g.actual_glucose - g.baseline_prediction).abs().mean())}
                   for h, g in results.groupby("horizon_minutes")}}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = BACKEND / "data/5min-glucose-data-prepared.csv"
    if not source.exists():
        source = BACKEND / "copies of old data/5min-glucose-data-prepared.csv"
    parser.add_argument("--data", type=Path, default=source)
    parser.add_argument("--output", type=Path, default=BACKEND / "artifacts/personalized-meal-regression")
    args = parser.parse_args()
    data = pd.read_csv(args.data, dtype={"Patient": str, "Recording": str}, low_memory=False)
    print(json.dumps(run(data, args.output), indent=2))


if __name__ == "__main__":
    main()
