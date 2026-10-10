"""Seven complete 24-hour blocks -> next-day mean CGM, using Ridge regression."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKEND = Path(__file__).resolve().parents[1]
META = ["Patient", "Recording", "forecast_start", "target_day", "target"]


def make_windows(data, min_samples=230):
    """Require >=80% of expected 288 readings in EACH input and target day."""
    data = data.copy()
    data["Date"] = pd.to_datetime(data.Date, errors="raise")
    data["glucose"] = pd.to_numeric(data["CGM (mg / dl)"], errors="coerce")
    candidates = [c for c in data if c.startswith(("food_", "insulin_", "med_", "csii_"))
                  and not c.endswith("normalized") and c not in ["food_status"]]
    # CBG is a separate measurement available during the input history only.
    # Also retain explicit no-food/history-completeness indicators.
    candidates += [c for c in ["CBG (mg / dl)", "explicit_no_food",
                    "history_120min_available", "minutes_since_logged_food",
                    "no_prior_logged_food"] if c in data]
    for col in candidates:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    examples, exclusions = [], []
    for (patient, recording), frame in data.groupby(["Patient", "Recording"]):
        frame = frame.sort_values("Date")
        if frame.Date.duplicated().any():
            raise ValueError(f"Duplicate timestamps: {recording}")
        anchor = frame.Date.min()
        frame["block"] = ((frame.Date - anchor).dt.total_seconds() // 86400).astype(int)
        daily = []
        for block, part in frame.groupby("block"):
            row = {"block": block, "count": part.glucose.count(), "mean": part.glucose.mean(),
                   "std": part.glucose.std(), "min": part.glucose.min(), "max": part.glucose.max()}
            for col in candidates:
                # Mean per 5-min record: consistent scale, not fabricated total drug exposure.
                row[col] = part[col].mean()
            daily.append(row)
        days = pd.DataFrame(daily).set_index("block")
        # A block is complete only if recording extends through its final 5-min slot.
        last_complete = int(((frame.Date.max() - anchor).total_seconds() + 300) // 86400) - 1
        eligible = 0
        for target in range(7, last_complete + 1):
            indexes = list(range(target - 7, target + 1))
            if not set(indexes).issubset(days.index) or (days.loc[indexes, "count"] < min_samples).any():
                continue
            history = days.loc[indexes[:-1]]
            row = {"Patient": patient, "Recording": recording,
                   "forecast_start": anchor + pd.Timedelta(days=target),
                   "target_day": target + 1, "target": days.loc[target, "mean"]}
            for offset, (_, day) in enumerate(history.iterrows(), start=1):
                for stat in ["mean", "std", "min", "max"]:
                    row[f"day{offset}_glucose_{stat}"] = day[stat]
            for col in candidates:
                row[f"history_{col}"] = history[col].mean()
            examples.append(row)
            eligible += 1
        exclusions.append({"Patient": patient, "Recording": recording,
                           "eligible_windows": eligible,
                           "has_day8": any(r["Recording"] == recording and r["target_day"] == 8 for r in examples),
                           "reason": "" if eligible else "Insufficient duration or CGM coverage"})
    if not examples:
        raise ValueError("No complete seven-day histories with next-day labels")
    return pd.DataFrame(examples), pd.DataFrame(exclusions)


def evaluate(frame, predictions):
    return {"r2": float(r2_score(frame.target, predictions)) if len(frame) > 1 else None,
            "mae": float(mean_absolute_error(frame.target, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(frame.target, predictions))),
            "day7_baseline_mae": float(mean_absolute_error(frame.target, frame.day7_glucose_mean)),
            "seven_day_baseline_mae": float(mean_absolute_error(frame.target,
                frame[[f"day{i}_glucose_mean" for i in range(1, 8)]].mean(axis=1)))}


def train(data, output):
    windows, excluded = make_windows(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Split people BEFORE fitting or selecting parameters: overlapping windows stay together.
    people = np.array(sorted(windows.Patient.unique()))
    if len(people) < 5:
        raise ValueError("Need at least five eligible patients for three distinct sets")
    np.random.default_rng(42).shuffle(people)
    first, second = int(len(people) * .6), int(len(people) * .8)
    assignments = {person: split for split, group in zip(
        ["train", "validation", "test"], [people[:first], people[first:second], people[second:]]) for person in group}
    windows["split"] = windows.Patient.map(assignments)
    features = [col for col in windows if col not in META + ["split"]]
    frames = {name: windows[windows.split == name] for name in ["train", "validation", "test"]}
    # Evaluate ALL chronological rolling examples in each held-out partition.
    # Inputs shift by one day: 1–7 -> 8, 2–8 -> 9, 3–9 -> 10, etc.
    best, best_score, best_alpha = None, float("inf"), None
    # Compare ordinary and regularized linear regression on validation only.
    for alpha in [0., 1., 10., 100., 1000.]:
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), LinearRegression() if alpha == 0 else Ridge(alpha=alpha))
        model.fit(frames["train"][features], frames["train"].target)
        score = mean_absolute_error(frames["validation"].target, model.predict(frames["validation"][features]))
        if score < best_score:
            best, best_score, best_alpha = model, score, alpha
    metrics = {name: evaluate(frame, best.predict(frame[features])) for name, frame in frames.items()}
    for name, frame in frames.items():
        result = frame.copy()
        result["predicted_glucose_mean"] = best.predict(frame[features])
        result["absolute_error"] = (result.target - result.predicted_glucose_mean).abs()
        result.to_csv(output / f"{name}.csv", index=False)
    rolling = windows.copy()
    rolling["predicted_glucose_mean"] = best.predict(rolling[features])
    rolling["absolute_error"] = (rolling.target - rolling.predicted_glucose_mean).abs()
    rolling.to_csv(output / "rolling_predictions.csv", index=False)
    patient_metrics = []
    for (patient, split), frame in rolling.groupby(["Patient", "split"]):
        patient_metrics.append({"Patient": patient, "split": split, "predictions": len(frame),
                                **evaluate(frame, frame.predicted_glucose_mean)})
    pd.DataFrame(patient_metrics).to_csv(output / "patient_metrics.csv", index=False)
    day8 = windows[windows.target_day == 8].copy()
    if not day8.empty:
        day8["predicted_glucose_mean"] = best.predict(day8[features])
        day8["absolute_error"] = (day8.target - day8.predicted_glucose_mean).abs()
        day8.to_csv(output / "day8_predictions.csv", index=False)
    excluded.to_csv(output / "recording_eligibility.csv", index=False)
    windows[[*META, "split"]].to_csv(output / "split_manifest.csv", index=False)
    coefficients = best[-1].coef_
    pd.DataFrame({"feature": features, "standardized_coefficient": coefficients}).assign(
        absolute_coefficient=np.abs(coefficients)).sort_values("absolute_coefficient", ascending=False).to_csv(output / "coefficients.csv", index=False)
    joblib.dump({"model": best, "features": features, "target": "next_24h_mean_cgm_mg_dl",
                 "alpha": best_alpha}, output / "linear_regression.joblib")
    report = {"target": "Next 24-hour mean CGM (mg/dL) from seven prior 24-hour blocks",
              "patient_split": "60/20/20, seed 42; no patient crosses splits",
              "selected_alpha": best_alpha, "metrics": metrics,
              "selected_model": type(best[-1]).__name__,
              "feature_count": len(features),
              "patients": {s: int(f.Patient.nunique()) for s, f in frames.items()},
              "examples": {s: len(f) for s, f in frames.items()},
              "day8_recordings": len(day8),
              "note": "All eligible rolling windows evaluated. Train predictions are in-sample. Test patients held out; validation selects the linear model. Each forecast uses only its preceding seven days."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = BACKEND / "data/5min-glucose-data-prepared.csv"
    if not default_data.exists():
        default_data = BACKEND / "copies of old data/5min-glucose-data-prepared.csv"
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=BACKEND / "artifacts/regression-ml-glucose")
    args = parser.parse_args()
    report = train(pd.read_csv(args.data, dtype={"Patient": str, "Recording": str}, low_memory=False), args.output)
    print(f"Selected linear model: {report['selected_model']} (alpha={report['selected_alpha']})")
    print("Label: next day's mean CGM in mg/dL. Seven-day rolling input history.")
    for split, scores in report["metrics"].items():
        print(f"{split.upper():10s} R2={scores['r2']:.4f}  MAE={scores['mae']:.2f} mg/dL  RMSE={scores['rmse']:.2f} mg/dL")
    print("R2 is a regression score, not percent accuracy. Only TEST is untouched evaluation.")
    print(f"Predictions and reports saved to {args.output}")


if __name__ == "__main__":
    main()
