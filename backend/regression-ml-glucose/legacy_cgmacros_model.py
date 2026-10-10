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

from backend.data import load_participants
from backend.features import FEATURES, TARGETS, build_meals


def train(meals, output):
    training = meals[meals.day.between(1, 7)]
    testing = meals[(meals.day == 8) & meals.participant.isin(training.participant)]
    if len(training) < 10 or testing.empty:
        raise ValueError("Need at least 10 usable days 1–7 meals and usable day 8 meals.")
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
        MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", max_depth=3,
            n_estimators=150, learning_rate=0.05, reg_lambda=5, random_state=42, n_jobs=1)))
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
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="backend/artifacts")
    parser.add_argument("--sensor", default="Dexcom GL", choices=["Dexcom GL", "Libre GL"])
    args = parser.parse_args()
    print(json.dumps(train(build_meals(load_participants(args.data), args.sensor), args.output), indent=2))


if __name__ == "__main__":
    main()
