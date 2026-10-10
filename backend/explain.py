"""Export individual SHAP contributions and global importance for one target."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from xgboost import DMatrix
from .data import DEFAULT_DATA, load_forecast_splits


def tree_contributions(estimator, x):
    # XGBoost's native TreeSHAP supports its current model serialization.
    # The last column is the expected value, not a feature contribution.
    return estimator.get_booster().predict(DMatrix(x), pred_contribs=True)[:, :-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="backend/artifacts/regression.joblib")
    parser.add_argument("--meals", default="backend/reports/meals.csv")
    parser.add_argument("--target", default=None)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", default="backend/reports")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    if bundle.get("dataset") == "daily_forecast":
        if args.target not in (None, "target"):
            raise ValueError("Daily forecast models explain --target target")
        splits, _ = load_forecast_splits(args.data)
        frame = splits[args.split]
        pipeline = bundle["model"]
        x = pipeline.steps[0][1].transform(frame[bundle["features"]])
        contributions = pd.DataFrame(tree_contributions(pipeline.steps[1][1], x),
                                     columns=bundle["features"])
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        contributions.abs().mean().sort_values(ascending=False).rename("mean_absolute_shap").to_csv(output / "importance_target.csv")
        for col in reversed(["Patient", "Recording", "forecast_start"]):
            contributions.insert(0, col, frame[col].to_numpy())
        contributions.to_csv(output / f"contributions_{args.split}_target.csv", index=False)
        return
    args.target = args.target or "glucose_60"
    meals = pd.read_csv(args.meals)
    meals = meals[meals.day == 8]
    if meals.empty:
        raise ValueError("No day 8 meals to explain")
    pipeline = bundle["model"]
    x = pipeline.steps[0][1].transform(meals[bundle["features"]])
    estimator = pipeline.steps[1][1].estimators_[bundle["targets"].index(args.target)]
    contributions = pd.DataFrame(tree_contributions(estimator, x), columns=bundle["features"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contributions.abs().mean().sort_values(ascending=False).rename("mean_absolute_shap").to_csv(output / f"importance_{args.target}.csv")
    contributions.insert(0, "Timestamp", meals.Timestamp.to_numpy())
    contributions.insert(0, "participant", meals.participant.to_numpy())
    contributions.to_csv(output / f"contributions_{args.target}.csv", index=False)


if __name__ == "__main__":
    main()
