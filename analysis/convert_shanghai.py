"""Convert the Shanghai T1DM/T2DM CGM dataset from Excel into tidy CSV files.

Outputs (in data/shanghai_csv/ by default):
    patients.csv     one row per CGM recording period, clinical info from the *_Summary.xlsx sheets
    glucose.csv      CGM readings (15 min) plus finger-stick blood glucose (CBG) and blood ketone
    meals.csv        dietary intake events (English and original Chinese descriptions)
    medications.csv  insulin and oral agent events in long format (one row per drug administration)

Usage:
    uv run python analysis/convert_shanghai.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
"""

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-recording sheets share column order, but header text varies between files.
RECORD_COLUMNS = [
    "timestamp",
    "cgm_mg_dl",
    "cbg_mg_dl",
    "blood_ketone_mmol_l",
    "diet_en",
    "diet_zh",
    "insulin_sc",
    "oral_agents",
    "csii_bolus",
    "csii_basal",
    "insulin_iv",
]

SUMMARY_COLUMNS = [
    "recording_id",
    "gender",
    "age_years",
    "height_m",
    "weight_kg",
    "bmi_kg_m2",
    "smoking_pack_years",
    "alcohol",
    "diabetes_type",
    "diabetes_duration_years",
    "acute_complications",
    "macrovascular_complications",
    "microvascular_complications",
    "comorbidities",
    "hypoglycemic_agents",
    "other_agents",
    "fasting_plasma_glucose_mg_dl",
    "postprandial_2h_plasma_glucose_mg_dl",
    "fasting_c_peptide_nmol_l",
    "postprandial_2h_c_peptide_nmol_l",
    "fasting_insulin_pmol_l",
    "postprandial_2h_insulin_pmol_l",
    "hba1c_mmol_mol",
    "glycated_albumin_pct",
    "total_cholesterol_mmol_l",
    "triglyceride_mmol_l",
    "hdl_cholesterol_mmol_l",
    "ldl_cholesterol_mmol_l",
    "creatinine_umol_l",
    "egfr_ml_min_1_73m2",
    "uric_acid_mmol_l",
    "blood_urea_nitrogen_mmol_l",
    "hypoglycemia",
]

NUMERIC_SUMMARY_COLUMNS = SUMMARY_COLUMNS[SUMMARY_COLUMNS.index("fasting_plasma_glucose_mg_dl") : -1]

MEAL_NOT_RECORDED = {"data not available", "未记录"}

INSULIN_SC_RE = re.compile(r"^(?P<drug>.+?),?\s+(?P<dose>[\d.]+)\s*IU$")
ORAL_AGENT_RE = re.compile(r"^(?P<drug>.+?)\s+(?P<dose>[\d.]+)\s*(?P<unit>mg|g)$")
INSULIN_IV_RE = re.compile(r"(?P<dose>[\d.]+)\s*IU\s+(?P<drug>[^,]+)")


def find_dataset_dir(data_dir: Path) -> Path:
    matches = [p.parent for p in data_dir.rglob("Shanghai_T1DM_Summary.xlsx")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one Shanghai dataset under {data_dir}, found {len(matches)}")
    return matches[0]


def split_recording_id(ids: pd.Series) -> pd.DataFrame:
    parts = ids.str.extract(r"^(?P<patient_id>\d+)_(?P<visit>\d+)_(?P<start_date>\d{8})$")
    if parts.isna().any().any():
        raise ValueError(f"Unexpected recording ids: {ids[parts.isna().any(axis=1)].tolist()}")
    return pd.DataFrame(
        {
            "recording_id": ids,
            "patient_id": parts["patient_id"].astype(int),
            "visit": parts["visit"].astype(int),
            "start_date": pd.to_datetime(parts["start_date"], format="%Y%m%d").dt.date,
        }
    )


def load_patients(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(dataset_dir.glob("Shanghai_*_Summary.xlsx")):
        df = pd.read_excel(path)
        df.columns = SUMMARY_COLUMNS
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df[NUMERIC_SUMMARY_COLUMNS] = df[NUMERIC_SUMMARY_COLUMNS].replace("/", pd.NA).apply(pd.to_numeric)
    df["gender"] = df["gender"].map({1: "female", 2: "male"})
    for col in ["alcohol", "diabetes_type", "hypoglycemia"]:
        df[col] = df[col].str.strip()

    ids = split_recording_id(df["recording_id"])
    return ids.merge(df, on="recording_id")


def load_recordings(dataset_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(dataset_dir.glob("Shanghai_*DM/*.xls*")):
        if path.name.startswith("~$"):  # Excel lock files
            continue
        df = pd.read_excel(path)
        if len(df.columns) != len(RECORD_COLUMNS):
            raise ValueError(f"{path.name}: expected {len(RECORD_COLUMNS)} columns, got {len(df.columns)}")
        df.columns = RECORD_COLUMNS
        df.insert(0, "recording_id", path.stem)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["timestamp"])

    misplaced = df["csii_bolus"].astype(str).str.strip().str.match(ORAL_AGENT_RE) & df["oral_agents"].isna()
    df.loc[misplaced, "oral_agents"] = df.loc[misplaced, "csii_bolus"]
    df.loc[misplaced, "csii_bolus"] = pd.NA

    ids = split_recording_id(df["recording_id"])[["patient_id", "visit"]]
    df.insert(1, "patient_id", ids["patient_id"].values)
    df.insert(2, "visit", ids["visit"].values)
    return df.sort_values(["recording_id", "timestamp"]).reset_index(drop=True)


def build_glucose(rec: pd.DataFrame) -> pd.DataFrame:
    cols = ["recording_id", "patient_id", "visit", "timestamp", "cgm_mg_dl", "cbg_mg_dl", "blood_ketone_mmol_l"]
    return rec[cols].dropna(subset=["cgm_mg_dl", "cbg_mg_dl", "blood_ketone_mmol_l"], how="all")


def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace(r"\s*\n\s*", "; ", regex=True)
        .str.strip(" ;")
        .replace("", pd.NA)
    )


def build_meals(rec: pd.DataFrame) -> pd.DataFrame:
    meals = rec.loc[
        rec["diet_en"].notna() | rec["diet_zh"].notna(),
        ["recording_id", "patient_id", "visit", "timestamp", "diet_en", "diet_zh"],
    ].copy()
    meals["diet_en"] = clean_text(meals["diet_en"])
    meals["diet_zh"] = clean_text(meals["diet_zh"])
    meals = meals.dropna(subset=["diet_en", "diet_zh"], how="all")
    not_recorded = meals["diet_en"].str.lower().isin(MEAL_NOT_RECORDED) | meals["diet_zh"].isin(MEAL_NOT_RECORDED)
    meals["description_recorded"] = ~not_recorded.fillna(False)
    return meals.rename(columns={"diet_en": "description_en", "diet_zh": "description_zh"})


def _events(rec: pd.DataFrame, column: str) -> pd.DataFrame:
    ev = rec.loc[rec[column].notna(), ["recording_id", "patient_id", "visit", "timestamp", column]]
    ev = ev.rename(columns={column: "description"})
    ev["description"] = ev["description"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return ev


def _apply_regex(ev: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    parsed = ev["description"].str.extract(pattern)
    ev["drug"] = parsed["drug"].str.strip()
    ev["dose"] = pd.to_numeric(parsed["dose"])
    return ev


def build_medications(rec: pd.DataFrame) -> pd.DataFrame:
    sc = _apply_regex(_events(rec, "insulin_sc"), INSULIN_SC_RE)
    sc["route"] = "insulin_sc"
    sc["unit"] = sc["dose"].notna().map({True: "IU", False: None})

    oral = _events(rec, "oral_agents")
    # Some cells are written "acarbose, 50 mg", so only split on commas not followed by a dose.
    oral["description"] = oral["description"].str.split(r",(?!\s*\d)")
    oral = oral.explode("description")
    oral["description"] = oral["description"].str.strip()
    oral = oral[oral["description"] != ""]
    parsed = oral["description"].str.extract(ORAL_AGENT_RE)
    oral["drug"] = parsed["drug"].str.strip()
    oral["dose"] = pd.to_numeric(parsed["dose"])
    oral["unit"] = parsed["unit"]
    oral["route"] = "oral"

    pump = []
    for column, route, unit in [("csii_bolus", "csii_bolus", "IU"), ("csii_basal", "csii_basal", "IU/h")]:
        ev = _events(rec, column)
        ev["dose"] = pd.to_numeric(ev["description"], errors="coerce")
        ev["drug"] = ev["dose"].notna().map({True: "Novolin R", False: None})
        ev["unit"] = ev["dose"].notna().map({True: unit, False: None})
        ev["route"] = route
        pump.append(ev)

    iv = _events(rec, "insulin_iv")
    iv = iv[~iv["description"].str.startswith("CSII")]  # stray header text in one file
    iv = _apply_regex(iv, INSULIN_IV_RE)
    iv["unit"] = iv["dose"].notna().map({True: "IU", False: None})
    iv["route"] = "insulin_iv"

    meds = pd.concat([sc, oral, *pump, iv], ignore_index=True)
    cols = ["recording_id", "patient_id", "visit", "timestamp", "route", "drug", "dose", "unit", "description"]
    return meds[cols].sort_values(["recording_id", "timestamp", "route"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "shanghai_csv")
    args = parser.parse_args()

    dataset_dir = find_dataset_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = load_patients(dataset_dir)
    rec = load_recordings(dataset_dir)

    missing = set(rec["recording_id"]) ^ set(patients["recording_id"])
    if missing:
        print(f"Warning: recordings not present in both summary and time series: {sorted(missing)}")

    outputs = {
        "patients.csv": patients,
        "glucose.csv": build_glucose(rec),
        "meals.csv": build_meals(rec),
        "medications.csv": build_medications(rec),
    }
    for name, df in outputs.items():
        path = args.out_dir / name
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}: {len(df):,} rows")


if __name__ == "__main__":
    main()
