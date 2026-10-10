"""Per-patient calibration from the first seven days, and the replay simulation of confirmed meals.

* Spike thresholds: how far this patient's glucose really rose after their own logged meals
  (75th percentile = spike, 90th = high), instead of one static number for everyone.
* Carb response: this patient's median mg/dL rise per gram of carbohydrate in those meals.
  The trained model gives sweet foods no weight (the training diets had almost none), so the
  sugar in a meal is added on top of the model's forecast at that personal rate, labelled as such.
* Simulation: a confirmed meal's own effect (forecast with the meal minus forecast without it)
  is added to the recorded CGM for three hours, so the replay shows what eating it would do.
"""
import numpy as np
import pandas as pd

from .catalog import NUTRITION
from .prepare import NO_FOOD, UNKNOWN
from .spike import items

FALLBACK = {"rise": 50.0, "severe_rise": 80.0}
DEFAULT_MG_DL_PER_G = 2.0
MIN_MEALS = 5
SUGAR_SHAPE = [0.6, 1.0, 0.75, 0.45]    # share of the sugar effect at +30/60/90/120 min; simple sugars act fast
SUGAR_CAP = 150.0
EFFECT_MINUTES = [0, 30, 60, 90, 120, 180]
NUTRIENT_KEYS = sorted(NUTRITION, key=len, reverse=True)


def nutrients(name):
    """(carbs, sugar) per gram for the longest known food name inside `name`, else None."""
    text = name.casefold()
    for key in NUTRIENT_KEYS:
        if key in text:
            carbs, sugar = NUTRITION[key]
            return carbs / 100, sugar / 100
    return None


def meal_carbs(text):
    """Estimated grams of carbohydrate in a free-text meal, or None if any amount is unknown."""
    total = 0.0
    for item in items(text):
        per_gram = nutrients(item["name"])
        if item["amount"] is None or per_gram is None:
            return None
        total += item["amount"] * per_gram[0]
    return total


def sugar_grams(text, foods=None):
    """Sugar in the meal text. USDA numbers sent by the meal builder win over the built-in table,
    scaled by the amount in the text (so 'half portion' variants scale too)."""
    sent = {}
    for food in foods or []:
        if food.get("sugar_g") is not None and food.get("grams"):
            sent[food["name"].strip().casefold()] = food["sugar_g"] / food["grams"]
    total, parts = 0.0, []
    for item in items(text):
        if item["amount"] is None:
            continue
        name = item["name"].strip().casefold()
        per_gram = sent.get(name)
        source = "usda"
        if per_gram is None:
            known = nutrients(name)
            per_gram, source = (known[1], "table") if known else (0.0, "unknown")
        grams = item["amount"] * per_gram
        total += grams
        if grams > 0:
            parts.append({"name": item["name"], "sugar_g": round(grams, 1), "source": source})
    return total, parts


def first_week_meals(raw):
    """Real rise after each logged meal in days 1-7 that had no other meal within two hours."""
    start = raw.Date.min()
    week = raw[raw.Date < start + pd.Timedelta(days=7)]
    cgm = week.set_index("Date")["CGM (mg / dl)"].astype(float).dropna()
    text = week["Dietary intake"].fillna("").astype(str).str.strip()
    lowered = text.str.casefold()
    meals = week[(lowered != "") & ~lowered.isin(NO_FOOD) & ~lowered.isin(UNKNOWN)]
    times = list(meals.Date)
    out = []
    for i, (t, food) in enumerate(zip(meals.Date, text[meals.index])):
        if i + 1 < len(times) and times[i + 1] - t < pd.Timedelta(minutes=120):
            continue
        before = cgm.loc[t - pd.Timedelta(minutes=10):t]
        after = cgm.loc[t:t + pd.Timedelta(minutes=150)]
        if before.empty or len(after) < 20:
            continue
        rise = float(after.max() - before.iloc[-1])
        out.append({"time": t, "food": food, "rise": rise, "carbs": meal_carbs(food)})
    return out


def profile(raw):
    meals = first_week_meals(raw)
    rises = np.array([m["rise"] for m in meals])
    if len(rises) >= MIN_MEALS:
        thresholds = {"rise": round(max(20.0, float(np.percentile(rises, 75)))),
                      "severe_rise": round(max(30.0, float(np.percentile(rises, 90)))), "personal": True}
    else:
        thresholds = {**FALLBACK, "personal": False}
    thresholds["meals"] = len(rises)
    ratios = [m["rise"] / m["carbs"] for m in meals if m["carbs"] and m["carbs"] >= 15]
    if len(ratios) >= MIN_MEALS:
        carb = {"mg_dl_per_g": round(float(np.clip(np.median(ratios), 0.5, 5.0)), 2), "personal": True}
    else:
        carb = {"mg_dl_per_g": DEFAULT_MG_DL_PER_G, "personal": False}
    carb["meals"] = len(ratios)
    return {"thresholds": thresholds, "carb_response": carb,
            "typical_rise_mg_dl": round(float(np.median(rises))) if len(rises) else None}


def add_sugar(result, text, foods, personal):
    """Add the sugar effect the model can't see to a forecast (a copy); records how it was computed."""
    sugar, parts = sugar_grams(text, foods)
    rate = personal["carb_response"]["mg_dl_per_g"]
    peak = min(SUGAR_CAP, sugar * rate)
    effect = [peak * share for share in SUGAR_SHAPE]
    out = {**result, "predictions": [], "feature_contributions": list(result["feature_contributions"])}
    for p, extra in zip(result["predictions"], effect):
        p = {**p, "glucose_mg_dl": p["glucose_mg_dl"] + extra, "change_mg_dl": p["change_mg_dl"] + extra}
        if extra > 0:
            top = p["top_features"] + [{"feature": "sugar_adjustment", "contribution_mg_dl": extra}]
            p["top_features"] = sorted(top, key=lambda e: abs(e["contribution_mg_dl"]), reverse=True)[:3]
            out["feature_contributions"].append({"feature": "sugar_adjustment", "horizon_minutes": p["minutes_after_meal"],
                                                 "contribution": extra})
        out["predictions"].append(p)
    out["sugar_adjustment"] = {"sugar_g": round(sugar, 1), "mg_dl_per_g": rate, "personal": personal["carb_response"]["personal"],
                               "mg_dl": [round(e, 1) for e in effect], "items": parts}
    return out


def meal_effect(with_meal, without_meal):
    """The meal's own effect at +30/60/90/120: forecast with it minus the same moment with no food."""
    return [round(a["change_mg_dl"] - b["change_mg_dl"], 1)
            for a, b in zip(with_meal["predictions"], without_meal["predictions"])]


def simulate(raw, meals):
    """Recorded CGM plus each confirmed meal's effect for three hours. Adds a `recorded` column."""
    sim = raw.copy()
    sim["recorded"] = pd.to_numeric(sim["CGM (mg / dl)"], errors="coerce").astype(float)
    glucose = sim["recorded"].copy()
    for meal in meals:
        effect = meal.get("meal_effect")
        if not effect:
            continue
        t0 = pd.Timestamp(meal["meal_time"])
        minutes = (sim.Date - t0).dt.total_seconds() / 60
        window = (minutes > 0) & (minutes <= EFFECT_MINUTES[-1])
        glucose[window] += np.interp(minutes[window], EFFECT_MINUTES, [0, *effect, 0])
    sim["CGM (mg / dl)"] = glucose.clip(lower=40).round(1)
    return sim


def with_meals(raw, meals):
    """History for the model: confirmed meals and their medication written in as if logged."""
    out = raw.copy()
    for meal in meals:
        t0 = pd.Timestamp(meal["meal_time"])
        row = out.Date == t0
        if not row.any():
            continue
        if not out.loc[row, "Dietary intake"].fillna("").astype(str).str.strip().iloc[0]:
            out.loc[row, "Dietary intake"] = meal["dietary_intake"]
        for column, text in (meal.get("medication") or {}).items():
            if column in out:
                out.loc[row, column] = text
    return out
