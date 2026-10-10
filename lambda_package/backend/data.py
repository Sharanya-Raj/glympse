"""Load extracted CGMacros participant CSVs without joining by row position."""
from pathlib import Path

import pandas as pd

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
