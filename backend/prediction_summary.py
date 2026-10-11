"""Read-only evidence audit and grouped explanations; never changes forecasts."""
import math
import re
from collections import defaultdict

import pandas as pd

MEDICATION_COLUMNS = ["Insulin dose - s.c.", "Non-insulin hypoglycemic agents",
    "CSII - bolus insulin (Novolin R, IU)", "CSII - basal insulin (Novolin R, IU / H)"]
NO_MEAL = {"", "no food ingested", "none", "nan", "na", "n/a"}


def nonempty(frame, column):
    return frame[column].fillna("").astype(str).str.strip() if column in frame else pd.Series("", index=frame.index)


def group_name(feature):
    if feature == "baseline": return "Baseline glucose"
    if feature.startswith("premeal_"): return "Recent glucose trend"
    if feature in {"hour_sin", "hour_cos"}: return "Time of day"
    if feature.startswith(("insulin_", "med_", "medication_", "csii_")): return "Medications"
    if feature.startswith("food_logs_"): return "Meal history"
    if feature.startswith(("food_amount_", "food_quantity_")): return "Meal portion"
    if feature == "sugar_adjustment": return "Sugar adjustment"
    if feature.startswith("food_"): return "Meal type"
    return "Other model inputs"


def explain_peak(result):
    peak = max(result["predictions"], key=lambda row: row["glucose_mg_dl"])
    horizon = peak["minutes_after_meal"]
    grouped = defaultdict(float)
    members = defaultdict(list)
    for row in result.get("feature_contributions", []):
        if row["horizon_minutes"] != horizon:
            continue
        value = float(row["contribution"])
        if not math.isfinite(value): continue
        category = group_name(row["feature"])
        grouped[category] += value
        members[category].append({"feature": row["feature"], "contribution_mg_dl": value})
    ranked = sorted(grouped, key=lambda name: (-abs(grouped[name]), name))
    return {"horizon_minutes": horizon, "method": "absolute_net_group_contribution_at_sampled_peak",
        "reference": "Model reference inputs, not the pre-meal baseline; contributions are not causes or prediction accuracy.",
        "top_factors": [{"label": name, "contribution_mg_dl": grouped[name],
            "direction": "higher" if grouped[name] > 0 else "lower", "features": members[name]}
            for name in ranked if abs(grouped[name]) > 1e-8][:3],
        "not_modeled": ["Activity"],
        "note": "Sugar adjustment is an existing post-model rule, not a Ridge feature. The intercept and reference are not ranked."}


def evidence_summary(result, raw, recent, text):
    cutoff = pd.Timestamp(result["personal_cutoff"])
    start = cutoff - pd.Timedelta(days=7)
    # Match the model's seven completed 24-hour blocks, not calendar dates.
    daily = []
    for index in range(7):
        left = start + pd.Timedelta(days=index)
        rows = raw[(raw.Date >= left) & (raw.Date < left + pd.Timedelta(days=1))]
        cgm = pd.to_numeric(rows["CGM (mg / dl)"], errors="coerce")
        valid = rows.loc[cgm.notna() & cgm.gt(0) & cgm.lt(float("inf")), "Date"]
        occupied = int(((valid - left).dt.total_seconds() // 300).nunique())
        meal_text = nonempty(rows, "Dietary intake").str.lower()
        meals = int((~meal_text.isin(NO_MEAL)).sum())
        medication_entries = sum(int(nonempty(rows, col).ne("").sum()) for col in MEDICATION_COLUMNS)
        daily.append({"start": left.isoformat(), "end": (left + pd.Timedelta(days=1)).isoformat(),
            "cgm_slots_present": occupied, "cgm_slots_expected": 288,
            "cgm_coverage_percent": round(100 * occupied / 288, 1),
            "recorded_meal_events": meals, "medication_entries": medication_entries,
            "medication_completeness": "unknown", "meal_log_completeness": "unknown"})
    slots = sum(day["cgm_slots_present"] for day in daily)
    meal_time = pd.Timestamp(result["meal_time"])
    recent_slots = set()
    for row in recent:
        stamp = pd.to_datetime(row["timestamp"], utc=True).tz_convert(None)
        if meal_time - pd.Timedelta(minutes=30) <= stamp <= meal_time:
            recent_slots.add(int((stamp - (meal_time-pd.Timedelta(minutes=30))).total_seconds() // 300))
    portions = bool(re.search(r"\d+(?:\.\d+)?\s*(?:g|ml)\b", text, flags=re.I)) or text.strip().lower() == "no food ingested"
    checks = [
        {"category": "Seven-day CGM", "gap": slots / 2016 < .9,
         "detail": f"{slots} of 2016 five-minute slots contain uploaded CGM (90% evidence threshold)."},
        {"category": "Recent CGM", "gap": len(recent_slots) < 7,
         "detail": f"{len(recent_slots)} of 7 pre-meal five-minute slots supplied; these may include replay-simulated values."},
        {"category": "Personal response history", "gap": result["personal_training_meals"] < 5,
         "detail": f"{result['personal_training_meals']} completed meal responses; personal correction requires at least 5."},
        {"category": "Meal portion", "gap": not portions,
         "detail": "A g/ml portion is detectable; amounts for all ingredients are not independently verified." if portions else "No parseable g/ml portion; the original food inputs may be incomplete."},
        {"category": "Medication completeness", "gap": True,
         "detail": "The current event log cannot confirm a complete medication regimen or explicitly verified non-use. Blank entries are unknown, not confirmed no medication."}]
    score = sum(int(check["gap"]) for check in checks)
    return {"version": "evidence-gaps-v1", "gap_count": score, "checks_count": len(checks),
        "label": f"Evidence gaps: {score} / {len(checks)}", "calibrated": False,
        "meaning": "Unweighted count of evidence checks that are incomplete. This is not a probability of error, confidence percentage, or prediction interval.",
        "checks": checks, "daily": daily, "history_start": start.isoformat(), "history_end": cutoff.isoformat(),
        "cgm_coverage_percent": round(100 * slots / 2016, 1),
        "limitations": ["Meal-free CGM rows are not missing meal nutrition; unrecorded meals cannot be detected.",
            "Uploaded CGM coverage does not verify measurement provenance; pre-interpolated uploads cannot be identified.",
            "Activity is not an input to this model.",
            "The original preprocessing remains unchanged and encodes some blank food/medication fields as zero indicators; this audit does not correct that limitation.",
            "Prediction intervals and statistical confidence require held-out measured-glucose calibration."]}


def summarize_prediction(result, raw, recent, text):
    return {**result, "uncertainty": evidence_summary(result, raw, recent, text), "explanation": explain_peak(result)}
