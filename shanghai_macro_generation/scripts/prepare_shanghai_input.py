"""Build the pipeline input (meal_id, meal_text, ...) from the converted Shanghai meals table.

Source: data/shanghai_csv/meals.csv (written by analysis/convert_shanghai.py)
    recording_id, patient_id, visit, timestamp, description_en, description_zh, description_recorded

Example:
    python scripts/prepare_shanghai_input.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT.parent / "data" / "shanghai_csv" / "meals.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "shanghai_meals.csv"


def prepare(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source)
    recorded = df["description_recorded"].astype(str).str.lower().eq("true")
    df = df[recorded & df["description_en"].notna()].copy()
    timestamps = pd.to_datetime(df["timestamp"])
    df["meal_id"] = df["recording_id"].astype(str) + "_" + timestamps.dt.strftime("%Y%m%dT%H%M")
    df["meal_text"] = df["description_en"].astype(str).str.strip()
    out = df[["meal_id", "meal_text", "patient_id", "recording_id", "visit", "timestamp"]]
    if out["meal_id"].duplicated().any():
        raise ValueError("Derived meal_id is not unique; check the source timestamps")
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)

    out = prepare(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(out)} meals ({out['meal_text'].nunique()} unique texts) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
