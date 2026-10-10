"""Convert one raw patient workbook (backend/data/*.xls[x]) into an upload-ready 5-minute CSV.

The raw recordings sample CGM every 15 minutes; the model and upload rules use a 5-minute
grid (>=230 readings/day, >=4 readings in the 30 minutes before a meal). CGM is linearly
interpolated onto that grid only between real readings at most MAX_GAP apart; longer gaps
stay blank. Meal and treatment text is kept at its original timestamps, never copied.

    python -m backend.ml.export_patient backend/data/2000_0_20201230.xlsx -o patient_2000.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from .service import COLUMNS

MAX_GAP = pd.Timedelta(minutes=30)
STEP = pd.Timedelta(minutes=5)


def to_five_minute(raw):
    data = raw.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    data = data.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="first")
    data = data.set_index("Date")
    cgm = pd.to_numeric(data["CGM (mg / dl)"], errors="coerce").dropna()
    grid = pd.date_range(data.index.min(), data.index.max(), freq=STEP)
    index = grid.union(data.index)
    out = data.reindex(index)
    filled = cgm.reindex(index.union(cgm.index)).interpolate(method="time", limit_area="inside").reindex(index)
    # Blank interpolated points whose surrounding real readings are too far apart.
    times = pd.Series(index, index=index)
    previous = pd.Series(cgm.index, index=cgm.index).reindex(index, method="ffill")
    following = pd.Series(cgm.index, index=cgm.index).reindex(index, method="bfill")
    real = times.isin(cgm.index)
    filled[~real & ((following - previous) > MAX_GAP)] = float("nan")
    out["CGM (mg / dl)"] = filled.round(1)
    out.index.name = "Date"
    return out.reset_index()


def export(source, output):
    source = Path(source)
    raw = pd.read_excel(source)
    patient, recording = source.stem.split("_")[0], source.stem
    result = to_five_minute(raw)
    result["Patient"], result["Recording"] = patient, recording
    for col in COLUMNS:
        if col not in result:
            result[col] = pd.NA
    result = result[COLUMNS]
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="raw patient workbook, e.g. backend/data/2000_0_20201230.xlsx")
    parser.add_argument("-o", "--output", type=Path, help="CSV path (default: backend/data/upload/<name>_5min.csv)")
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parents[1] / "data/upload" / f"{args.source.stem}_5min.csv"
    result = export(args.source, output)
    print(f"Saved {len(result):,} rows ({result['CGM (mg / dl)'].notna().sum():,} with CGM) to {output}")


if __name__ == "__main__":
    main()
