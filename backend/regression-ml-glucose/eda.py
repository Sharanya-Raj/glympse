"""Create a column-filtered glucose CSV copy and basic EDA reports.

Run from the repository root:
    python backend/regression-ml-glucose/eda.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
COLUMNS = [
    "Recording", "Patient", "Date", "CGM (mg / dl)", "CBG (mg / dl)",
    "Dietary intake", "Insulin dose - s.c.", "Non-insulin hypoglycemic agents",
    "CSII - bolus insulin (Novolin R, IU)",
    "CSII - basal insulin (Novolin R, IU / H)",
]


def analyze(source, cleaned, output):
    source, cleaned, output = Path(source), Path(cleaned), Path(output)
    if source.resolve() == cleaned.resolve():
        raise ValueError("The cleaned copy must have a different path from the original.")
    # Read as strings so IDs, medication descriptions, and blank cells survive.
    original = pd.read_csv(source, dtype="string")
    missing = set(COLUMNS) - set(original.columns)
    if missing:
        raise ValueError(f"Missing requested columns: {sorted(missing)}")
    data = original.loc[:, COLUMNS].copy()
    # Project assumption: an absent food log means no food at this timestamp.
    food_blank = data["Dietary intake"].fillna("").str.strip().eq("")
    data.loc[food_blank, "Dietary intake"] = "No food ingested"
    cleaned.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cleaned, index=False)
    output.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(data["Date"], errors="coerce")
    numeric_columns = ["CGM (mg / dl)", "CBG (mg / dl)",
                       "CSII - bolus insulin (Novolin R, IU)",
                       "CSII - basal insulin (Novolin R, IU / H)"]
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe().T
    summary["non_numeric_nonblank"] = (data[numeric_columns].notna() & numeric.isna()).sum()
    missingness = pd.DataFrame({"missing_count": data.isna().sum(),
                                "missing_percent": data.isna().mean() * 100})
    overview = {"original_shape": list(original.shape), "cleaned_shape": list(data.shape),
                "patients": int(data.Patient.nunique()),
                "recordings": int(data.Recording.nunique()),
                "date_start": str(dates.min()), "date_end": str(dates.max()),
                "invalid_or_missing_dates": int(dates.isna().sum()),
                "duplicate_rows": int(data.duplicated().sum()),
                "duplicate_recording_dates": int(data.duplicated(["Recording", "Date"]).sum())}
    patient_summary = data.assign(parsed_date=dates).groupby("Patient").agg(
        rows=("Recording", "size"), recordings=("Recording", "nunique"),
        first_date=("parsed_date", "min"), last_date=("parsed_date", "max"))
    data.head(10).to_csv(output / "preview.csv", index=False)
    summary.to_csv(output / "numeric_summary.csv")
    missingness.to_csv(output / "missingness.csv")
    patient_summary.to_csv(output / "patient_summary.csv")
    data.describe(include="all").T.to_csv(output / "all_columns_summary.csv")
    (output / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    report = "\n".join([
        "GLUCOSE DATA: BASIC EDA", json.dumps(overview, indent=2),
        "\nFirst five rows:", data.head().to_string(index=False),
        "\nNumeric statistics (sample standard deviation):", summary.to_string(),
        "\nMissing values:", missingness.to_string(),
        "\nNotes: Original file unchanged. Only columns were removed; no rows were dropped.",
        "Blank dietary entries mean No food ingested by project assumption.",
        "Other missing values were not replaced with zero. Text treatments and dietary intake",
        "are summarized categorically; patient/recording identifiers are not measurements.",
    ])
    (output / "eda_report.txt").write_text(report, encoding="utf-8")
    # Avoid printing patient-level records into the terminal by default.
    print(json.dumps(overview, indent=2))
    print("\nNumeric statistics:\n" + summary.to_string())
    print(f"\nCleaned copy: {cleaned}\nEDA reports, including preview: {output}")
    return overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BACKEND / "data/5min-glucose-data.csv")
    parser.add_argument("--cleaned", type=Path, default=BACKEND / "data/5min-glucose-data-cleaned.csv")
    parser.add_argument("--output", type=Path, default=BACKEND / "reports/regression-ml-glucose")
    args = parser.parse_args()
    analyze(args.input, args.cleaned, args.output)


if __name__ == "__main__":
    main()
