"""Spike risk from a meal forecast, and model-predicted what-if variants of the planned meal.

Levels use the forecast peak (any horizon) and its rise from the pre-meal baseline. Drivers
are the model's additive food contributions at the peak horizon, not proven causes.
"""
import re

from .normalize import FOODS

HIGH = 180            # mg/dL, top of the usual target range
RISE = 50             # mg/dL rise from baseline that counts as a spike
SEVERE = 250
SEVERE_RISE = 80
ITEM = re.compile(r"([^\d]+?)\s*(\d+(?:\.\d+)?)\s*(g|ml)\b", re.IGNORECASE)


def assess(result, dietary_intake):
    base = result["baseline_glucose"]
    peak = max(result["predictions"], key=lambda p: p["glucose_mg_dl"])
    value, rise = peak["glucose_mg_dl"], peak["glucose_mg_dl"] - base
    level = ("high" if value > SEVERE or rise > SEVERE_RISE
             else "moderate" if value > HIGH or rise > RISE else "none")
    horizon = peak["minutes_after_meal"]
    # Contributions are relative to an average meal, so absent foods also "contribute"; keep only this meal's.
    text = dietary_intake.lower()
    present = {f"food_{kind}" for kind, pattern in FOODS.items() if re.search(pattern, text)}
    units = {item["unit"] for item in items(dietary_intake)}
    present |= {name for unit, name in [("g", "food_amount_grams"), ("ml", "food_amount_ml")] if unit in units}
    drivers = sorted((e for e in result["feature_contributions"] if e["horizon_minutes"] == horizon
                      and e["feature"] in present and e["contribution"] > 0),
                     key=lambda e: e["contribution"], reverse=True)[:3]
    return {"level": level, "peak_glucose_mg_dl": value, "peak_minutes_after_meal": horizon,
            "rise_mg_dl": rise, "thresholds": {"high": HIGH, "rise": RISE, "severe": SEVERE, "severe_rise": SEVERE_RISE},
            "food_drivers": [{"feature": e["feature"], "contribution_mg_dl": e["contribution"]} for e in drivers]}


def items(text):
    """Split free text like 'Rice 100 g Egg 50 g, cake' into named items with optional amounts."""
    out = []
    for chunk in re.split(r"[,;+\n]|\band\b", text):
        position = 0
        for match in ITEM.finditer(chunk):
            out.append({"name": match[1].strip(" .-"), "amount": float(match[2]), "unit": match[3].lower()})
            position = match.end()
        rest = chunk[position:].strip(" .-")
        if re.search(r"[^\W\d_]", rest):
            out.append({"name": rest, "amount": None, "unit": None})
    return [item for item in out if item["name"]]


def is_kind(item, kind):
    return re.search(FOODS[kind], item["name"].lower()) is not None


def describe(meal):
    return ", ".join(f"{i['name']} {i['amount']:g} {i['unit']}" if i["amount"] is not None else i["name"] for i in meal)


def halved(meal, kinds=None):
    return [{**i, "amount": i["amount"] / 2} if i["amount"] is not None and
            (kinds is None or any(is_kind(i, k) for k in kinds)) else i for i in meal]


def variants(dietary_intake):
    """Candidate changes to the planned meal, each as new dietary text. Only distinct, non-empty ones."""
    meal = items(dietary_intake)
    sweet = [i for i in meal if is_kind(i, "sweet")]
    options = []
    if sweet and len(sweet) < len(meal):
        options.append(("Skip the sweet items", [i for i in meal if not is_kind(i, "sweet")]))
    if any(i["amount"] is not None and (is_kind(i, "sweet") or is_kind(i, "starch")) for i in meal):
        options.append(("Half portion of sweet and starchy foods", halved(meal, ["sweet", "starch"])))
    if any(i["amount"] is not None for i in meal):
        options.append(("Half the whole meal", halved(meal)))
    seen, out = {describe(meal)}, []
    for label, changed in options:
        text = describe(changed)
        if text and text not in seen:
            seen.add(text)
            out.append({"label": label, "dietary_intake": text})
    return {"sweet_items": [i["name"] for i in sweet], "variants": out}
