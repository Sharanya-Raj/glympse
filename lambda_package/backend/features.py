"""Timestamp-aligned meal examples; future glucose is used only for labels."""
import numpy as np
import pandas as pd

FEATURES = ["Carbs", "Protein", "Fat", "Fiber", "Calories", "baseline",
            "premeal_slope", "recent_hr", "recent_activity", "hour_sin", "hour_cos",
            "history_mean", "history_std"]
TARGETS = ["glucose_30", "glucose_60", "glucose_90", "glucose_120",
           "peak", "time_to_peak", "excursion"]


def build_meals(data, sensor="Dexcom GL", max_gap_minutes=15):
    if sensor not in data:
        raise ValueError(f"Sensor column unavailable: {sensor}")
    rows = []
    for participant, frame in data.groupby("participant"):
        frame = frame.sort_values("Timestamp").set_index("Timestamp")
        glucose = frame[sensor].dropna()
        glucose = glucose[glucose.between(40, 400)]
        meals = frame[frame["Meal Type"].astype(str).str.strip().str.lower().isin(
            ["breakfast", "lunch", "dinner", "snack"])]
        for time, meal in meals.iterrows():
            end = time + pd.Timedelta(minutes=120)
            # A second meal would confound the target response.
            if ((meals.index > time) & (meals.index <= end)).any():
                continue
            past = glucose.loc[time - pd.Timedelta(days=7):time]
            if past.empty or (time - past.index[-1]).total_seconds() > 600:
                continue
            baseline = float(past.iloc[-1])
            grid = pd.date_range(time, end, freq="5min")
            values = []
            for point in grid:
                if point == time:
                    values.append(baseline)
                    continue
                left = glucose.loc[:point]
                right = glucose.loc[point:]
                if left.empty or right.empty:
                    break
                before, after = left.index[-1], right.index[0]
                gap = (after - before).total_seconds() / 60
                if gap > max_gap_minutes:
                    break
                fraction = (point - before).total_seconds() / (gap * 60) if gap else 0
                values.append(float(left.iloc[-1] + fraction * (right.iloc[0] - left.iloc[-1])))
            if len(values) != len(grid):
                continue
            recent = frame.loc[time - pd.Timedelta(minutes=30):time]
            pre = past.loc[time - pd.Timedelta(minutes=30):]
            elapsed = (pre.index[-1] - pre.index[0]).total_seconds() / 60
            hour = time.hour + time.minute / 60
            row = {"participant": participant, "Timestamp": time, "day": int(meal.day),
                   **{col: meal.get(col, np.nan) for col in FEATURES[:5]},
                   "baseline": baseline,
                   "premeal_slope": (pre.iloc[-1] - pre.iloc[0]) / elapsed if elapsed else np.nan,
                   "recent_hr": recent.HR.mean() if "HR" in recent else np.nan,
                   "recent_activity": recent.Mets.mean() if "Mets" in recent else np.nan,
                   "hour_sin": np.sin(2 * np.pi * hour / 24),
                   "hour_cos": np.cos(2 * np.pi * hour / 24),
                   "history_mean": past.mean(), "history_std": past.std()}
            row.update({f"glucose_{minute}": values[minute // 5] for minute in [30, 60, 90, 120]})
            row.update(peak=max(values), time_to_peak=int(np.argmax(values) * 5),
                       excursion=max(values) - baseline)
            rows.append(row)
    return pd.DataFrame(rows, columns=["participant", "Timestamp", "day", *FEATURES, *TARGETS])
