"""Seven complete 24-hour blocks -> next-day mean CGM, using Ridge regression."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    return {"mae": float(mean_absolute_error(frame.target, predictions)),
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
    # Validate on each person's earliest eligible prediction, including day 8 when present.
    for name in ["validation", "test"]:
        frames[name] = frames[name].sort_values("target_day").groupby(["Patient", "Recording"]).head(1)
    best, best_score, best_alpha = None, float("inf"), None
    for alpha in [1., 10., 100., 1000.]:
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=alpha))
        model.fit(frames["train"][features], frames["train"].target)
        score = mean_absolute_error(frames["validation"].target, model.predict(frames["validation"][features]))
        if score < best_score:
            best, best_score, best_alpha = model, score, alpha
    metrics = {name: evaluate(frame, best.predict(frame[features])) for name, frame in frames.items()}
    for name, frame in frames.items():
        result = frame.copy()
        result["predicted_glucose_mean"] = best.predict(frame[features])
        result.to_csv(output / f"{name}.csv", index=False)
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
              "patients": {s: int(f.Patient.nunique()) for s, f in frames.items()},
              "examples": {s: len(f) for s, f in frames.items()},
              "day8_recordings": len(day8),
              "note": "Train predictions are in-sample. Test is held out; validation selects alpha. No day8 food or medications used."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BACKEND / "data/5min-glucose-data-prepared.csv")
    parser.add_argument("--output", type=Path, default=BACKEND / "artifacts/regression-ml-glucose")
    args = parser.parse_args()
    print(json.dumps(train(pd.read_csv(args.data, dtype={"Patient": str, "Recording": str}, low_memory=False), args.output), indent=2))


if __name__ == "__main__":
    main()
