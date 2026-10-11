"""Extract one patient's app-ready CGM CSV locally, without AWS access.

Default: Patient 2088 from data/T2DM_5min_with_macros.csv ->
data/sharanya_cg/patient_2088.csv. Chinese-named columns, CBG and blood
ketones are omitted. Other source columns, including macros, are retained.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import io
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
# Match backend.ml.service.COLUMNS, except the explicitly excluded CBG column.
# The backend supplies a blank optional CBG field when it reads this file.
APP_COLUMNS = [
    "Recording", "Patient", "Date", "CGM (mg / dl)", "Dietary intake",
    "Insulin dose - s.c.", "Non-insulin hypoglycemic agents",
    "CSII - bolus insulin (Novolin R, IU)",
    "CSII - basal insulin (Novolin R, IU / H)",
]
EXCLUDED = {"CBG (mg / dl)", "Blood Ketone (mmol / L)"}
CHINESE_HEADER = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def prepare_patient(source, output, patient="2088", recording=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Output must be different from the source dataset")
    rows, timestamps = [], set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("Source CSV has duplicate column names")
        required = {"Patient", "Recording", "Date", "CGM (mg / dl)"}
        if missing := required - set(headers):
            raise ValueError(f"Source is missing required columns: {sorted(missing)}")
        excluded = [name for name in headers if name in EXCLUDED or CHINESE_HEADER.search(name)]
        columns = APP_COLUMNS + [name for name in headers if name not in APP_COLUMNS and name not in excluded]
        for line, row in enumerate(reader, start=2):
            if (row.get("Patient") or "").strip() != str(patient):
                continue
            if recording is not None and row["Recording"] != recording:
                continue
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row at line {line}")
            stamp = datetime.fromisoformat(row["Date"].strip())
            # The app uses UTC for explicit offsets and preserves naive clock times.
            if stamp.tzinfo is not None:
                stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
            if stamp in timestamps:
                raise ValueError(f"Duplicate timestamp for this patient at line {line}")
            timestamps.add(stamp)
            glucose = row["CGM (mg / dl)"].strip()
            if glucose:
                try:
                    value = Decimal(glucose)
                except InvalidOperation as exc:
                    raise ValueError(f"Invalid CGM value at line {line}") from exc
                if not value.is_finite() or value <= 0:
                    raise ValueError(f"CGM must be positive and finite at line {line}")
            cleaned = {name: row.get(name, "") for name in columns}
            cleaned["Date"] = stamp.isoformat(sep=" ")
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"No rows found for Patient {patient}" + (f", Recording {recording}" if recording else ""))
    recordings = {row["Recording"] for row in rows}
    if len(recordings) != 1 or not next(iter(recordings)).strip():
        raise ValueError(f"Select one nonempty Recording with --recording; found: {sorted(recordings)}")
    if len(rows) > 100000:
        raise ValueError("The app accepts at most 100,000 rows per upload")
    rows.sort(key=lambda row: row["Date"])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    if output.exists() and output.read_bytes() != data:
        raise ValueError(f"Output already contains different data; choose another --output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return {"output": str(output), "patient": str(patient), "rows": len(rows),
            "columns": len(columns), "excluded": excluded,
            "first_timestamp": rows[0]["Date"], "last_timestamp": rows[-1]["Date"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/T2DM_5min_with_macros.csv")
    parser.add_argument("--patient", default="2088", help="Exact source Patient ID")
    parser.add_argument("--recording", help="Select one recording if this patient has multiple")
    parser.add_argument("--output", type=Path, help="Default: data/sharanya_cg/patient_<ID>.csv")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.patient):
        parser.error("Patient ID may contain only letters, digits, underscores and hyphens")
    output = args.output or ROOT / "data/sharanya_cg" / f"patient_{args.patient}.csv"
    try:
        result = prepare_patient(args.source, output, args.patient, args.recording)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {result['rows']:,} rows, {result['columns']} columns to {result['output']}")
    print(f"Patient {result['patient']}: {result['first_timestamp']} through {result['last_timestamp']}")
    print("Excluded: " + ", ".join(ascii(name) for name in result["excluded"]))


if __name__ == "__main__":
    main()
