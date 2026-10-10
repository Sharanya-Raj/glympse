"""Explicit adapters for the supplied Shanghai and CGMacros files."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUTRIENTS = ["carbs", "protein", "fat", "calories", "fiber"]
MED_COLUMNS = {
    "Insulin dose - s.c.": ("sc", None),
    "Non-insulin hypoglycemic agents": ("oral", None),
    "CSII - bolus insulin (Novolin R, IU)": ("csii_bolus", "IU"),
    "CSII - basal insulin (Novolin R, IU / H)": ("csii_basal", "IU/H"),
    "Insulin dose - i.v.": ("iv", None),
}


@dataclass
class Patient:
    participant: str
    source: str
    cohort: str
    glucose: pd.DataFrame
    meals: pd.DataFrame
    activity: pd.DataFrame
    medication: pd.DataFrame


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()
    return frame


def require(frame, columns, path):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def times(frame, column):
    result = pd.to_datetime(frame[column], errors="coerce")
    if result.isna().any():
        raise ValueError(f"Invalid timestamps in {column}")
    return result


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def glucose_frame(frame, time, value, interpolated=None):
    result = pd.DataFrame({"time": time, "glucose": numeric(frame, value),
                           "interpolated": False if interpolated is None else interpolated})
    if "Recording" in frame:
        result["recording"] = frame["Recording"].astype(str)
    result = result.dropna(subset=["glucose"]).sort_values("time")
    if (result.glucose <= 0).any() or not np.isfinite(result.glucose).all():
        raise ValueError("Glucose must be finite positive mg/dL")
    if result.time.duplicated().any():
        # Identical duplicated rows may come from overlapping recording exports.
        if result.groupby("time").glucose.nunique().max() > 1:
            raise ValueError("Conflicting glucose values at the same patient timestamp")
        result = result.drop_duplicates("time")
    return result.reset_index(drop=True)


def medication_events(frame):
    records = []
    pattern = re.compile(r"(.+?)\s*,?\s+(\d+(?:\.\d+)?)\s*(IU|mg|g)\b", re.I)
    for column, (route, fixed_unit) in MED_COLUMNS.items():
        if column not in frame:
            continue
        for _, row in frame.loc[frame[column].notna()].iterrows():
            text = str(row[column]).strip()
            if not text or text in {"/", "nan"}:
                continue
            if fixed_unit:
                dose = pd.to_numeric(text, errors="coerce")
                matches = [("novolin r", dose, fixed_unit)] if pd.notna(dose) else []
            else:
                # Require a unit; preserve g versus mg rather than infer corrections.
                matches = [(m.group(1).strip(" ,;").lower(), float(m.group(2)), m.group(3).lower())
                           for m in pattern.finditer(text)]
            for drug, dose, unit in matches:
                records.append({"time": row["time"], "drug": drug, "route": route,
                                "unit": unit, "dose": dose, "raw": text, "parsed": True})
            if not matches:
                records.append({"time": row["time"], "drug": "unparsed", "route": route,
                                "unit": "unknown", "dose": np.nan, "raw": text, "parsed": False})
    return pd.DataFrame(records, columns=["time", "drug", "route", "unit", "dose", "raw", "parsed"])


def load_sources(t2dm_path, cg_root):
    """Load available sources so split membership is independent of selected run."""
    patients, files, inventory = [], [], []
    cg_root = Path(cg_root)
    for path in sorted(cg_root.rglob("*")):
        if path.is_file():
            inventory.append({"path": str(path), "bytes": path.stat().st_size})
    bio_path = cg_root / "bio.csv"
    bio = read_csv(bio_path) if cg_root.exists() else pd.DataFrame(columns=["subject", "A1c PDL (Lab)"])
    require(bio, ["subject", "A1c PDL (Lab)"], bio_path)
    if bio.subject.duplicated().any():
        raise ValueError("Duplicate subjects in CGMacros bio.csv")
    cohorts = {}
    for _, row in bio.iterrows():
        a1c = pd.to_numeric(row["A1c PDL (Lab)"], errors="coerce")
        cohorts[int(row.subject)] = "unknown" if pd.isna(a1c) else ("healthy" if a1c < 5.7 else "prediabetes" if a1c < 6.5 else "t2dm")
    if bio_path.exists():
        files.append(bio_path)
    seen = set()
    for path in sorted(cg_root.rglob("CGMacros-*.csv")):
        match = re.fullmatch(r"CGMacros-(\d+)", path.stem)
        if not match:
            continue
        subject = int(match.group(1))
        if subject in seen:
            raise ValueError(f"Duplicate CGMacros participant export: {subject}")
        seen.add(subject)
        frame = read_csv(path)
        require(frame, ["Timestamp", "Dexcom GL", "Meal Type", "Carbs", "Protein", "Fat", "Calories"], path)
        time = times(frame, "Timestamp")
        meals = pd.DataFrame({"time": time, "meal_type": frame["Meal Type"],
                              **{name: numeric(frame, name.title()) for name in NUTRIENTS}})
        meals = meals.loc[frame["Meal Type"].notna()].copy()
        meals["nutrition_complete"] = True
        meals["macro_confidence"] = np.nan
        meals["meal_id"] = path.stem + "|" + meals.time.astype(str)
        meals["recording"] = path.stem
        meals["amount_consumed"] = numeric(frame, "Amount Consumed").loc[meals.index]
        activity = pd.DataFrame({"time": time, "mets": numeric(frame, "METs") / 10,
                                 "hr": numeric(frame, "HR"), "activity_calories": numeric(frame, "Calories (Activity)")})
        patients.append(Patient(f"cg_macros:{subject:03}", "cg_macros", cohorts.get(subject, "unknown"),
                                glucose_frame(frame, time, "Dexcom GL"), meals, activity,
                                pd.DataFrame(columns=["time", "drug", "route", "unit", "dose", "raw", "parsed"])))
        files.append(path)
    if not seen and cg_root.exists():
        raise ValueError(f"No CGMacros participant CSVs in {cg_root}")
    path = Path(t2dm_path)
    if not path.exists():
        return finish_metadata(patients, files, inventory)
    frame = read_csv(path)
    require(frame, ["Patient", "Recording", "Date", "CGM (mg / dl)", "Interpolated", "meal_id",
                    "Dietary intake", "macro_status", "meal_carbs_g", "meal_protein_g", "meal_fat_g", "meal_calories_kcal"], path)
    if frame.Patient.isna().any() or frame.Recording.isna().any():
        raise ValueError("T2DM patient and recording identities are required")
    frame["time"] = times(frame, "Date")
    flags = frame.Interpolated.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Unknown T2DM interpolation flags")
    frame["_interpolated"] = flags.eq("true")
    for subject, group in frame.groupby("Patient", sort=True):
        participant = f"t2dm:{subject}"
        # Keep all recorded food events, including meals with missing nutrition,
        # so they still exclude overlapping response windows.
        text = group["Dietary intake"].fillna("").astype(str).str.strip()
        food = group.meal_id.notna() | (~text.str.lower().isin(["", "/", "data not available", "未记录"]))
        meals = pd.DataFrame({"time": group.time, "meal_id": group.meal_id,
                              "recording": group.Recording, "meal_type": "unknown",
                              "nutrition_complete": group.macro_status.eq("ok"),
                              "macro_confidence": numeric(group, "macro_confidence"),
                              "carbs": numeric(group, "meal_carbs_g"), "protein": numeric(group, "meal_protein_g"),
                              "fat": numeric(group, "meal_fat_g"), "calories": numeric(group, "meal_calories_kcal"),
                              "fiber": np.nan}).loc[food].copy()
        meals["meal_id"] = meals.meal_id.fillna(participant + "|" + meals.time.astype(str))
        patients.append(Patient(participant, "t2dm", "t2dm",
                                glucose_frame(group, group.time, "CGM (mg / dl)", group._interpolated),
                                meals, pd.DataFrame(columns=["time", "mets", "hr", "activity_calories"]),
                                medication_events(group)))
    files.append(path)
    return finish_metadata(patients, files, inventory)


def finish_metadata(patients, files, inventory):
    if not patients:
        raise ValueError("Neither source contains participant data")
    # Detect identical whole glucose exports even under different IDs/sources.
    signatures = {}
    for patient in patients:
        signature = hashlib.sha256(pd.util.hash_pandas_object(patient.glucose[["time", "glucose"]], index=False).values.tobytes()).hexdigest()
        if signature in signatures and not patient.glucose.empty:
            raise ValueError(f"Duplicate glucose export: {patient.participant} and {signatures[signature]}")
        signatures[signature] = patient.participant
    metadata = {"files": [{"path": str(p), "sha256": fingerprint(p)} for p in files],
                "cg_macros_inventory": inventory,
                "provenance": {"t2dm": "5-minute supplied table: flagged interpolation from coarser CGM; history uses unflagged measurements",
                               "cg_macros": "Dense 1-minute Dexcom values with no original-observation flags; history withheld by 5 minutes as a conservative interpolation guard"},
                "nutrition": "Planned/recorded meal macros. Actual Amount Consumed is retained for auditing only, never used as a prediction feature.",
                "macro_confidence": {"t2dm": "Read from macro_confidence; missing values remain unknown",
                                     "cg_macros": "No confidence field supplied; unscored, not an observed confidence of 1"}}
    return patients, metadata
