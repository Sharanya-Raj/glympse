"""Run with python -m backend.eda --data PATH --output backend/reports."""
import argparse
from pathlib import Path
import pandas as pd

from .data import DEFAULT_DATA, is_forecast_dataset, load_forecast_splits, load_participants
from .features import build_meals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default="backend/reports")
    parser.add_argument("--sensor", default="Dexcom GL", choices=["Dexcom GL", "Libre GL"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if is_forecast_dataset(args.data):
        splits, features = load_forecast_splits(args.data)
        data = pd.concat(splits.values(), ignore_index=True)
        data.to_csv(output / "forecasts.csv", index=False)
        data[features + ["target"]].isna().mean().rename("missing_fraction").to_csv(output / "missingness.csv")
        data.groupby("split").agg(rows=("target", "size"), patients=("Patient", "nunique"),
            recordings=("Recording", "nunique"), target_mean=("target", "mean")).to_csv(output / "coverage.csv")
        data[features + ["target"]].describe().to_csv(output / "forecast_summary.csv")
        print(f"{len(data)} daily forecasts; {len(features)} features. Reports: {output}")
        return
    data = load_participants(args.data)
    data.isna().mean().rename("missing_fraction").to_csv(output / "missingness.csv")
    data.groupby("participant").agg(rows=("Timestamp", "size"),
        observed_days=("day", "nunique"), first=("Timestamp", "min"),
        last=("Timestamp", "max")).to_csv(output / "coverage.csv")
    data.groupby(["participant", "day"])[args.sensor].agg(["count", "mean", "std", "min", "max"]).to_csv(output / "daily_glucose.csv")
    meals = build_meals(data, args.sensor)
    meals.to_csv(output / "meals.csv", index=False)
    meals.groupby(["participant", "day"]).size().rename("usable_meals").to_csv(output / "meal_counts.csv")
    meals.select_dtypes("number").describe().to_csv(output / "meal_summary.csv")
    print(f"{data.participant.nunique()} participants; {len(meals)} usable meals. Reports: {output}")


if __name__ == "__main__":
    main()
