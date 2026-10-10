"""Export individual SHAP contributions and global importance for one target."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import shap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="backend/artifacts/regression.joblib")
    parser.add_argument("--meals", default="backend/reports/meals.csv")
    parser.add_argument("--target", default="glucose_60")
    parser.add_argument("--output", default="backend/reports")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    meals = pd.read_csv(args.meals)
    meals = meals[meals.day == 8]
    if meals.empty:
        raise ValueError("No day 8 meals to explain")
    pipeline = bundle["model"]
    x = pipeline.steps[0][1].transform(meals[bundle["features"]])
    estimator = pipeline.steps[1][1].estimators_[bundle["targets"].index(args.target)]
    contributions = pd.DataFrame(shap.TreeExplainer(estimator).shap_values(x), columns=bundle["features"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contributions.abs().mean().sort_values(ascending=False).rename("mean_absolute_shap").to_csv(output / f"importance_{args.target}.csv")
    contributions.insert(0, "Timestamp", meals.Timestamp.to_numpy())
    contributions.insert(0, "participant", meals.participant.to_numpy())
    contributions.to_csv(output / f"contributions_{args.target}.csv", index=False)


if __name__ == "__main__":
    main()
