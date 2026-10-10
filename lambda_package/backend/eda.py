"""Run with python -m backend.eda --data PATH --output backend/reports."""
import argparse
from pathlib import Path

from .data import load_participants
from .features import build_meals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="backend/reports")
    parser.add_argument("--sensor", default="Dexcom GL", choices=["Dexcom GL", "Libre GL"])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
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
