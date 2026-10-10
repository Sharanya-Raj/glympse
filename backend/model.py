"""Train regression on days 1–7; evaluate day 8 for those participants."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from .data import DEFAULT_DATA, is_forecast_dataset, load_forecast_splits, load_participants
from .features import FEATURES, TARGETS, build_meals


def make_regressor(multioutput=False):
    estimator = XGBRegressor(objective="reg:squarederror", max_depth=3,
        n_estimators=150, learning_rate=0.05, reg_lambda=5, random_state=42, n_jobs=1)
    if multioutput:
        estimator = MultiOutputRegressor(estimator)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), estimator)


def train(meals, output):
    training = meals[meals.day.between(1, 7)]
    testing = meals[(meals.day == 8) & meals.participant.isin(training.participant)]
    if len(training) < 10 or testing.empty:
        raise ValueError("Need at least 10 usable days 1–7 meals and usable day 8 meals.")
    model = make_regressor(multioutput=True)
    model.fit(training[FEATURES], training[TARGETS])
    prediction = model.predict(testing[FEATURES])
    metrics = {}
    for index, target in enumerate(TARGETS):
        truth = testing[target]
        metrics[target] = {"mae": float(mean_absolute_error(truth, prediction[:, index])),
                           "rmse": float(np.sqrt(mean_squared_error(truth, prediction[:, index])))}
        if target.startswith("glucose_"):
            metrics[target]["baseline_mae"] = float(mean_absolute_error(truth, testing.baseline))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "targets": TARGETS}, output / "regression.joblib")
    report = {"protocol": "Known participants: calendar days 1–7 train, day 8 test",
              "train_meals": len(training), "test_meals": len(testing), "metrics": metrics}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    results = testing[["participant", "Timestamp", *TARGETS]].copy()
    for index, target in enumerate(TARGETS):
        results[f"predicted_{target}"] = prediction[:, index]
    results.to_csv(output / "day8_predictions.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default="backend/artifacts")
    parser.add_argument("--sensor", default="Dexcom GL", choices=["Dexcom GL", "Libre GL"])
    args = parser.parse_args()
    if is_forecast_dataset(args.data):
        splits, features = load_forecast_splits(args.data)
        report = train_forecasts(splits, features, args.output)
    else:
        report = train(build_meals(load_participants(args.data), args.sensor), args.output)
    print(json.dumps(report, indent=2))


def train_forecasts(splits, features, output):
    """Fit on supplied training rows and evaluate both held-out splits."""
    model = make_regressor()
    model.fit(splits["train"][features], splits["train"].target)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "targets": ["target"],
                 "dataset": "daily_forecast"}, output / "regression.joblib")
    report = {"protocol": "Supplied patient-disjoint train/validation/test splits; train-only fitting",
              "features": features, "train_rows": len(splits["train"]), "metrics": {}}
    for split in ("validation", "test"):
        frame = splits[split]
        prediction = model.predict(frame[features])
        metrics = {"rows": len(frame), "mae": float(mean_absolute_error(frame.target, prediction)),
                   "rmse": float(np.sqrt(mean_squared_error(frame.target, prediction)))}
        if "day7_glucose_mean" in frame:
            metrics["baseline_mae"] = float(mean_absolute_error(frame.target, frame.day7_glucose_mean))
        report["metrics"][split] = metrics
        results = frame[["Patient", "Recording", "forecast_start", "target_day", "target"]].copy()
        results["predicted_glucose_mean"] = prediction
        results["absolute_error"] = np.abs(frame.target - prediction)
        results.to_csv(output / f"{split}_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
