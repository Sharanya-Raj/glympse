"""Train one shared model for deployment; never run during signup or uploads."""
import argparse
import uuid
from pathlib import Path

import joblib
import pandas as pd

from .model import fit_population, meal_examples
from .service import prepared_history


def main():
    backend = Path(__file__).resolve().parent.parent
    # source = backend / "copies of old data/5min-glucose-data.csv"
    source = "C:\\Users\\omran\\code\\glyimpse\\data\\shanghai_csv\\Copy of T2DM_all_5min.xlsx - T2DM_5min.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=source)
    parser.add_argument("--output", type=Path, default=backend / "artifacts/population.joblib")
    args = parser.parse_args()
    raw = pd.read_csv(args.data, dtype="string")
    # Cohort IDs must stay distinct here; ownership replacement is for user uploads only.
    raw["Date"] = pd.to_datetime(raw.Date, errors="raise", utc=True).dt.tz_convert(None)
    data = prepared_history(raw)
    examples, features, _ = meal_examples(data)
    model, people = fit_population(examples, features, excluded_patient=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = uuid.uuid4().hex
    joblib.dump({"model": model, "features": features, "version": version,
                 "schema": 1, "training_patients": people}, args.output)
    print(f"Saved shared model {version} trained from {len(people)} patients to {args.output}")


if __name__ == "__main__":
    main()
