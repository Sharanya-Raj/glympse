"""Prepare explicit missing-food values and past-only meal-log features."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from normalize import enrich, audit

BACKEND = Path(__file__).resolve().parents[1]
UNKNOWN = {"", "na", "n/a", "nan", "none", "null", "data not available"}
NO_FOOD = {"no food", "no food ingested", "no intake", "fasting"}


def prepare(data):
    data = data.copy()
    text = data["Dietary intake"].fillna("").str.strip()
    normalized = text.str.casefold()
    no_food = normalized.isin(NO_FOOD) | normalized.eq("")
    unknown = normalized.isin(UNKNOWN) & ~no_food
    text = text.mask(no_food, "No food ingested")
    data["Dietary intake"] = text.mask(unknown, pd.NA)
    data["food_status"] = np.select([unknown, no_food],
                                     ["unknown", "explicit_no_food"], default="logged_food")
    data["food_logged_now"] = (~unknown & ~no_food).astype(int)
    data["food_entry_unknown"] = unknown.astype(int)
    data["explicit_no_food"] = no_food.astype(int)
    data["_timestamp"] = pd.to_datetime(data.Date, errors="raise")
    data["_order"] = np.arange(len(data))
    groups = []
    for _, group in data.groupby(["Patient", "Recording"], sort=False):
        group = group.sort_values("_timestamp").copy()
        if group._timestamp.duplicated().any():
            raise ValueError("Duplicate timestamps within a patient recording")
        events = pd.Series(group.food_logged_now.to_numpy(), index=group._timestamp)
        # Disjoint elapsed-time bands: [0,30], (30,60], (60,120] minutes.
        counts = {minutes: events.rolling(f"{minutes}min", closed="both").sum().to_numpy()
                  for minutes in (30, 60, 120)}
        group["food_logs_0_30min"] = counts[30].astype(int)
        group["food_logs_30_60min"] = (counts[60] - counts[30]).astype(int)
        group["food_logs_60_120min"] = (counts[120] - counts[60]).astype(int)
        elapsed = (group._timestamp - group._timestamp.iloc[0]).dt.total_seconds() / 60
        group["history_120min_available"] = (elapsed >= 120).astype(int)
        last_meal = group._timestamp.where(group.food_logged_now.eq(1)).ffill()
        group["minutes_since_logged_food"] = (group._timestamp - last_meal).dt.total_seconds() / 60
        group["no_prior_logged_food"] = last_meal.isna().astype(int)
        groups.append(group)
    return pd.concat(groups).sort_values("_order").drop(columns=["_timestamp", "_order"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BACKEND / "data/5min-glucose-data-cleaned.csv")
    parser.add_argument("--output", type=Path, default=BACKEND / "data/5min-glucose-data-prepared.csv")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Output must be a separate copy")
    data = pd.read_csv(args.input, dtype="string")
    result = enrich(prepare(data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, na_rep="NA")
    report = BACKEND / "reports/regression-ml-glucose"
    report.mkdir(parents=True, exist_ok=True)
    audit(result, report)
    print(f"Saved {len(result):,} rows to {args.output}")
    print(result.food_status.value_counts().to_string())


if __name__ == "__main__":
    main()
