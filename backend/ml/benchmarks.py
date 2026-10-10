"""Evaluate saved meal predictions; bootstrap uncertainty by patient, not row."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

MODELS = {"personalized": "predicted_glucose", "shared": "population_prediction",
          "persistence": "baseline_prediction", "personal_mean": "personal_mean_prediction"}


def scores(group, column):
    actual = group.actual_glucose.to_numpy()
    pred = group[column].to_numpy()
    errors = pred - actual
    baseline = group.baseline_prediction.to_numpy()
    valid_percentage = actual > 0
    relative_errors = np.abs(errors[valid_percentage]) / actual[valid_percentage]
    return {"forecasts": len(group), "patients": group.Patient.nunique(),
            "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "median_absolute_error": float(np.median(np.abs(errors))),
            "bias": float(errors.mean()),
            "r2": float(r2_score(actual, pred)) if len(actual) > 1 else np.nan,
            "change_r2": float(r2_score(actual - baseline, pred - baseline)) if len(actual) > 1 else np.nan,
            "mape_percent": float(100 * relative_errors.mean()) if len(relative_errors) else np.nan,
            "percentage_metric_count": int(valid_percentage.sum()),
            "within_15_percent_accuracy": float(100 * (relative_errors <= .15).mean()) if len(relative_errors) else np.nan,
            "within_10_mg_dl_percent": float(100 * (np.abs(errors) <= 10).mean()),
            "within_30_mg_dl_percent": float(100 * (np.abs(errors) <= 30).mean()),
            "within_20_mg_dl_percent": float(100 * (np.abs(errors) <= 20).mean())}


def interval(values):
    values = np.asarray(values)
    rng = np.random.default_rng(42)
    draws = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [.025, .975])


def benchmark(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = results.copy()
    if "personal_mean_prediction" not in results:
        raise ValueError("Run python -m backend.ml.evaluate to produce benchmark predictions")
    patient_rows, horizon_rows, day_rows, paired = [], [], [], []
    for (patient, horizon), group in results.groupby(["Patient", "horizon_minutes"]):
        for name, column in MODELS.items():
            patient_rows.append({"Patient": patient, "horizon_minutes": horizon,
                                 "model": name, **scores(group, column)})
    patient_table = pd.DataFrame(patient_rows)
    for horizon, group in results.groupby("horizon_minutes"):
        for name, column in MODELS.items():
            per_patient = patient_table[(patient_table.horizon_minutes == horizon) & (patient_table.model == name)]
            lower, upper = interval(per_patient.mae)
            horizon_rows.append({"horizon_minutes": horizon, "model": name,
                **scores(group, column), "patient_macro_mae": per_patient.mae.mean(),
                "macro_mae_ci_low": lower, "macro_mae_ci_high": upper})
        wide = patient_table[patient_table.horizon_minutes == horizon].pivot(index="Patient", columns="model", values="mae")
        for name in MODELS:
            if name == "personalized":
                continue
            improvements = wide[name] - wide.personalized
            low, high = interval(improvements)
            paired.append({"horizon_minutes": horizon, "comparator": name,
                "patient_macro_mae_improvement": improvements.mean(),
                "ci_low": low, "ci_high": high,
                "patients_improved_percent": 100 * (improvements > 0).mean()})
    for (day, horizon), group in results.groupby(["day", "horizon_minutes"]):
        for name, column in MODELS.items():
            day_rows.append({"day": day, "horizon_minutes": horizon, "model": name, **scores(group, column)})
    table = pd.DataFrame(horizon_rows)
    table.to_csv(output / "benchmark_by_horizon.csv", index=False)
    patient_table.to_csv(output / "benchmark_by_patient.csv", index=False)
    pd.DataFrame(day_rows).to_csv(output / "benchmark_by_day.csv", index=False)
    comparisons = pd.DataFrame(paired)
    comparisons.to_csv(output / "paired_improvements.csv", index=False)
    lines = ["# Meal-response benchmark report", "",
        "Exploratory leave-one-patient-out rolling evaluation. Parameters are fixed;",
        "this dataset has already informed development, so this is not a pristine external test.", "",
        "MAE/RMSE are mg/dL. R2 measures absolute glucose fit; change_R2 tests meal-related",
        "changes after removing pre-meal baseline. Within-20 is a descriptive tolerance rate,",
        "not classification accuracy or a clinical acceptance criterion.",
        "MAPE is mean absolute percentage error, NOT accuracy. Percentage metrics exclude",
        "nonpositive actual glucose (see percentage_metric_count). Accuracy within +/-15%",
        "means the fraction of predictions with absolute error <=15% of actual glucose.",
        "Do not report 100 minus MAPE as model accuracy. Thresholds are descriptive choices.", "",
        "| Minutes | Model | MAE | RMSE | R2 | Change R2 | MAPE | Accuracy within 20 mg/dL | Accuracy within 15% |",
        "|---|---|---|---|---|---|---|---|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.horizon_minutes} | {row.model} | {row.mae:.2f} | {row.rmse:.2f} | {row.r2:.3f} | {row.change_r2:.3f} | {row.mape_percent:.1f}% | {row.within_20_mg_dl_percent:.1f}% | {row.within_15_percent_accuracy:.1f}% |")
    lines += ["", "Comparators: shared model without personal adjustment; persistence (pre-meal",
        "glucose at every horizon); personal mean (baseline plus prior-seven-day mean meal",
        "change; population-free fallback to zero change if no history).", "",
        "Patient macro MAE gives equal weight to each patient. Its 95% bootstrap interval",
        "resamples patients (1,000 draws, seed 42). Paired improvement is comparator MAE",
        "minus personalized MAE; positive favors personalization. An interval crossing zero",
        "does not show a clear improvement in this sample. These intervals describe patient",
        "sampling variability, not per-prediction uncertainty or future deployment guarantees.", "",
        "Inspect benchmark_by_patient.csv for failures hidden by averages, benchmark_by_day.csv",
        "for day-8 versus later performance, and paired_improvements.csv for paired comparisons.",
        "Small patient samples produce unstable R2. Predictions at four horizons share the",
        "same meal; do not treat all rows as independent observations."]
    (output / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[1] / "artifacts/personalized-meal-regression"
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    results = pd.read_csv(args.output / "meal_predictions.csv", dtype={"Patient": str})
    print(benchmark(results, args.output).to_string(index=False))


if __name__ == "__main__":
    main()
