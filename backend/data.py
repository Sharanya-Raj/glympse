"""Load extracted CGMacros participant CSVs without joining by row position."""
from pathlib import Path
import re

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "sharanya_cg"


def is_forecast_dataset(root):
    return (Path(root) / "train.csv").is_file()


def load_forecast_splits(root):
    """Load prepared daily forecasts; only history columns are predictors."""
    splits = {}
    features = None
    for split in ("train", "validation", "test"):
        path = Path(root) / f"{split}.csv"
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        required = {"Patient", "Recording", "forecast_start", "target_day", "target"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        selected = [col for col in frame if re.fullmatch(r"day[1-7]_glucose_(mean|std|min|max)", col)
                    or col.startswith("history_")]
        if not selected or (features is not None and set(selected) != set(features)):
            raise ValueError(f"{path.name}: missing or inconsistent history features")
        features = selected if features is None else features
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f"{path.name}: empty dataset or missing forecast metadata/target")
        frame["forecast_start"] = pd.to_datetime(frame["forecast_start"], errors="raise")
        if frame.duplicated(["Patient", "Recording", "forecast_start"]).any():
            raise ValueError(f"{path.name}: duplicate forecasts")
        if "split" in frame and not frame["split"].eq(split).all():
            raise ValueError(f"{path.name}: split labels do not match filename")
        for col in [*features, "target", "target_day"]:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
            if np.isinf(frame[col]).any():
                raise ValueError(f"{path.name}: infinite values in {col}")
        frame["split"] = split
        splits[split] = frame
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(splits[left].Patient) & set(splits[right].Patient):
            raise ValueError(f"Patients overlap between {left} and {right}")
    return splits, features

NUMERIC = ["Dexcom GL", "Libre GL", "HR", "Calories (Activity)", "Mets",
           "Carbs", "Protein", "Fat", "Fiber", "Calories"]


def load_participants(root):
    files = sorted(Path(root).rglob("CGMacros-*.csv"))
    if not files:
        raise ValueError(f"No CGMacros participant CSVs found under {root}")
    frames = []
    seen = set()
    for path in files:
        if path.stem in seen:
            raise ValueError(f"Duplicate participant file: {path.stem}")
        seen.add(path.stem)
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        required = {"Timestamp", "Meal Type", "Carbs", "Protein", "Fat", "Fiber"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
        if frame["Timestamp"].isna().any():
            raise ValueError(f"{path.name}: invalid timestamps")
        if frame["Timestamp"].duplicated().any():
            raise ValueError(f"{path.name}: duplicate timestamps")
        for col in NUMERIC:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["participant"] = path.stem
        frame = frame.sort_values("Timestamp")
        # Calendar study days, not counts of observed days (gaps stay visible).
        frame["day"] = (frame.Timestamp.dt.normalize() - frame.Timestamp.min().normalize()).dt.days + 1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
