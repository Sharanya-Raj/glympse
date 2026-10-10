"""Attach per-meal macro estimates to the Shanghai T2DM 5-minute CGM table.

Every original row and column is kept unchanged (read as text, so formatting is preserved). Rows whose
(Recording, Date) match a processed meal get extra columns with the meal's macros, confidence and
provenance; all other rows leave those columns empty.

Example:
    python scripts/merge_with_cgm.py \
        --meals data/processed/usda_warmup/meals_with_macros.csv \
        --output ../data/T2DM_5min_with_macros.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CGM = REPO_ROOT / "data" / "shanghai_csv" / "Copy of T2DM_all_5min.xlsx - T2DM_5min.csv"
DEFAULT_MEALS = PROJECT_ROOT / "data" / "processed" / "meals_with_macros.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "T2DM_5min_with_macros.csv"

SOURCE_LABELS = {
    "manual_override": "manual override",
    "database_direct": "USDA direct match",
    "database_approximate": "USDA approximate match",
    "llm_estimate": "LLM estimate",
    "unresolved": "unresolved (no nutrition)",
}

MEAL_COLUMNS = {
    "meal_id": "meal_id",
    "calories": "meal_calories_kcal",
    "carbs_g": "meal_carbs_g",
    "protein_g": "meal_protein_g",
    "fat_g": "meal_fat_g",
    "total_grams": "meal_total_grams",
    "confidence": "macro_confidence",
    "status": "macro_status",
    "num_food_items": "macro_num_food_items",
    "num_db_direct": "macro_num_usda_direct",
    "num_db_approximate": "macro_num_usda_approximate",
    "num_llm_estimates": "macro_num_llm_estimates",
    "num_manual_overrides": "macro_num_manual_overrides",
    "num_unresolved_items": "macro_num_unresolved",
    "parse_warnings": "macro_parse_warnings",
}


def _minute_key(recording: pd.Series, timestamp: pd.Series) -> pd.Series:
    return recording.astype(str) + "|" + pd.to_datetime(timestamp).dt.strftime("%Y-%m-%d %H:%M")


def _fmt(value, digits: int = 0) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def describe_item(row: pd.Series) -> str:
    """One food item as 'text -> how its nutrition was obtained'."""
    source = row["nutrition_source"]
    label = SOURCE_LABELS.get(source, source)
    grams = _fmt(row["estimated_grams"])
    head = f"{row['normalized_food_name']} ({grams} g)" if grams else f"{row['normalized_food_name']} (amount unknown)"
    if source in ("database_direct", "database_approximate"):
        detail = f"{label} '{row['matched_food_name']}' (FDC {_fmt(row['fdc_id'])}, conf {row['nutrition_confidence']:.2f})"
    elif source == "llm_estimate":
        model = row.get("estimator_model") or "LLM"
        detail = f"{label} by {model} (conf {row['nutrition_confidence']:.2f})"
    else:
        detail = label
    return f"{head} -> {detail}"


def summarize_breakdown(breakdown: pd.DataFrame) -> pd.DataFrame:
    breakdown = breakdown.copy()
    breakdown["_resolved_g"] = breakdown["estimated_grams"].where(breakdown["nutrition_source"] != "unresolved", 0)
    breakdown["_desc"] = breakdown.apply(describe_item, axis=1)
    grouped = breakdown.groupby("meal_id", sort=False)
    out = pd.DataFrame({
        "macro_item_sources": grouped["_desc"].agg("; ".join),
        "_resolved_g": grouped["_resolved_g"].sum(min_count=1),
        "_total_g": grouped["estimated_grams"].sum(min_count=1),
    })
    out["macro_weight_coverage"] = (out["_resolved_g"] / out["_total_g"]).round(3)
    return out[["macro_weight_coverage", "macro_item_sources"]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cgm", type=Path, default=DEFAULT_CGM)
    p.add_argument("--meals", type=Path, default=DEFAULT_MEALS)
    p.add_argument("--breakdown", type=Path, help="defaults to food_breakdown.csv next to --meals")
    p.add_argument("--metadata", type=Path, help="defaults to run_metadata.json next to --meals")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)
    breakdown_path = args.breakdown or args.meals.with_name("food_breakdown.csv")
    metadata_path = args.metadata or args.meals.with_name("run_metadata.json")

    cgm = pd.read_csv(args.cgm, dtype=str, keep_default_na=False, encoding="utf-8")
    meals = pd.read_csv(args.meals)
    breakdown = pd.read_csv(breakdown_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    meals = meals.join(summarize_breakdown(breakdown), on="meal_id")
    count_cols = [c for c in MEAL_COLUMNS if c.startswith("num_")]
    meals[count_cols] = meals[count_cols].astype("Int64")
    options = metadata.get("options", {})
    parser = options.get("parser", "?")
    fallback_on = not options.get("skip_llm_fallback")
    run = f"{metadata.get('run_id', 'unknown')} (parser={parser}, llm_fallback={'on' if fallback_on else 'off'}"
    if parser == "llm" or fallback_on:
        run += f", llm={metadata.get('llm_provider', '?')}/{metadata.get('llm_model', '?')}"
    meals["macro_pipeline_run"] = run + ")"
    meals["_key"] = _minute_key(meals["recording_id"], meals["timestamp"])
    if meals["_key"].duplicated().any():
        raise ValueError("Two meals share the same recording and minute; cannot attach unambiguously")
    added = meals.rename(columns=MEAL_COLUMNS)
    added = added[["_key", *MEAL_COLUMNS.values(), "macro_weight_coverage", "macro_item_sources", "macro_pipeline_run"]]

    cgm["_key"] = _minute_key(cgm["Recording"], cgm["Date"])
    merged = cgm.merge(added, on="_key", how="left", validate="many_to_one").drop(columns="_key")

    matched = merged["meal_id"].notna().sum()
    cgm_recordings = set(cgm["Recording"])
    unmatched = meals[~meals["_key"].isin(set(cgm["_key"]))]
    other_cohort = (~unmatched["recording_id"].astype(str).isin(cgm_recordings)).sum()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(merged)} rows x {merged.shape[1]} columns to {args.output}")
    print(f"Meals attached: {matched} of {len(meals)} "
          f"({len(unmatched)} not in this CGM file, {other_cohort} of them from recordings it does not contain)")
    print("Attached meal statuses:", merged["macro_status"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
