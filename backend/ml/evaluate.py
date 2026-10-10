"""Leave-one-patient-out meal regression with rolling seven-day personal adaptation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .benchmarks import benchmark
from .model import meal_examples, fit_population, fit_personal, predict, personal_history, HORIZONS, TARGETS
from .explain import contributions, daily_top_three

BACKEND = Path(__file__).resolve().parents[1]


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
            pooled = model.predict(upcoming[features])
            adapted = predict(model, coef, intercept, upcoming[features])
            glucose_effects, reference = contributions(model, coef, features, upcoming[features])
            # A simple personalized comparator: average previously observed meal change.
            historical_delta = history[TARGETS].mean().to_numpy() if len(history) else np.zeros(4)
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
                    predictions[-1]["personal_mean_prediction"] = meal.baseline + historical_delta[j]
                    predictions[-1]["actual_change"] = meal[TARGETS[j]]
                    # Exact additive contributions in standardized model space.
                    for k, feature in enumerate(features):
                        explanations.append({"Patient": patient, "Recording": recording, "day": day,
                            "meal_time": meal.meal_time, "horizon_minutes": horizon, "feature": feature,
                            "contribution": glucose_effects[i, j, k]})
                    reconstructed = (reference + model[-1].intercept_[j] + intercept[j]
                                     + np.sum(glucose_effects[i, j]))
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
    daily_top_three(effects).to_csv(output / "daily_top_features.csv", index=False)
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
    benchmark(results, output)
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
    summary = run(data, args.output)
    print(f"Evaluated {summary['meal_forecasts']} meal forecasts across {summary['patients_evaluated']} patients.")
    print(pd.read_csv(args.output / "benchmark_by_horizon.csv")[[
        "horizon_minutes", "model", "mae", "rmse", "r2", "mape_percent",
        "within_20_mg_dl_percent", "within_15_percent_accuracy"]].to_string(index=False))
    print(f"\nFull report: {args.output / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
