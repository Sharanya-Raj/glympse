"""Parse USDA FoodData Central records into one stable nutrition shape, and scale by grams.

Built against real API responses (see scripts/build_food_map.py and the probes that
informed it). Two shapes are handled because the two endpoints disagree:

  /foods/search -> foods[].foodNutrients   flat:   {"nutrientNumber": "203", "value": 1.84, "unitName": "G"}
  /food/{fdcId} -> foodNutrients           nested: {"nutrient": {"number": "203", ...}, "amount": 1.84}

Portions live only in the detail record (foodPortions) and also come two ways:
  FNDDS     -> human label in "portionDescription" ("1 piece")
  SR Legacy -> "portionDescription" is null; label is "amount" + "modifier" ("1 cup")
"""

# USDA nutrient numbers -> our per-100 g field names. 208 = kcal, 203/204/205 = macros.
# Fiber (291) and sugar (269) are cached and returned but not shown in the UI yet.
NUTRIENTS = {"208": "kcal", "203": "protein_g", "204": "fat_g",
             "205": "carbs_g", "291": "fiber_g", "269": "sugar_g"}
FIELDS = ["kcal", "protein_g", "fat_g", "carbs_g", "fiber_g", "sugar_g"]
# Macros the UI treats as required; null (never 0) when USDA omits them.
REQUIRED = ["protein_g", "fat_g", "carbs_g"]
# Portion labels that carry no useful "compare to your plate" meaning.
JUNK_PORTIONS = {"quantity not specified", "not specified"}


def _nutrient_number(entry):
    number = entry.get("nutrientNumber")
    if number is None:
        number = (entry.get("nutrient") or {}).get("number")
    return str(number).strip() if number is not None else ""


def _nutrient_value(entry):
    # Search uses "value"; detail uses "amount". Group rows (e.g. "Proximates") have neither.
    value = entry.get("value")
    if value is None:
        value = entry.get("amount")
    return value


def extract_per_100g(food_nutrients):
    """Pull the six nutrients we track from either endpoint's nutrient list (per 100 g)."""
    per = {name: None for name in FIELDS}
    for entry in food_nutrients or []:
        field = NUTRIENTS.get(_nutrient_number(entry))
        if not field:
            continue
        value = _nutrient_value(entry)
        if value is not None:
            try:
                per[field] = float(value)
            except (TypeError, ValueError):
                pass
    return per


def _portion_label(portion):
    """A readable label like '1 cup' or '1 piece', or '' when the portion is unusable."""
    description = (portion.get("portionDescription") or "").strip()
    if description and description.lower() not in JUNK_PORTIONS:
        return description
    amount = portion.get("amount")
    unit = (portion.get("measureUnit") or {}).get("name") or ""
    if not unit or unit.lower() == "undetermined":
        unit = (portion.get("modifier") or "").strip()
    if amount and unit and unit.lower() not in JUNK_PORTIONS:
        return f"{float(amount):g} {unit}"
    return ""


def extract_portions(detail):
    """USDA household portions as [{label, grams}], ordered by USDA's sequenceNumber."""
    rows = []
    for portion in detail.get("foodPortions") or []:
        grams = portion.get("gramWeight")
        label = _portion_label(portion)
        if grams is None or not label:
            continue
        try:
            grams = round(float(grams), 1)
        except (TypeError, ValueError):
            continue
        if grams <= 0:
            continue
        rows.append({"label": label, "grams": grams,
                     "_seq": portion.get("sequenceNumber")})
    rows.sort(key=lambda row: (row["_seq"] is None, row["_seq"] if row["_seq"] is not None else 0))
    for row in rows:
        row.pop("_seq")
    # Drop exact duplicate (label, grams) pairs while keeping USDA order.
    seen, unique = set(), []
    for row in rows:
        key = (row["label"].lower(), row["grams"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _round_nutrient(field, value):
    return round(value) if field == "kcal" else round(value, 1)


def scale_nutrients(per_100g, grams):
    """Scale per-100 g values to `grams`. null passes through; everything rounds sensibly."""
    factor = grams / 100.0
    out = {}
    for field, value in per_100g.items():
        out[field] = None if value is None else _round_nutrient(field, value * factor)
    return out


def estimate_kcal(per_100g):
    """4*carbs + 4*protein + 9*fat when all three macros are known, else None."""
    carbs, protein, fat = per_100g.get("carbs_g"), per_100g.get("protein_g"), per_100g.get("fat_g")
    if None in (carbs, protein, fat):
        return None
    return round(4 * carbs + 4 * protein + 9 * fat)


def build_nutrition(fdc_id, description, data_type, food_nutrients, detail=None):
    """One stable record for the API and the on-disk cache.

    detail (the /food/{id} response) supplies portions; pass None when only a search
    record is available (then portions are empty and the UI falls back to per 100 g).
    """
    per = extract_per_100g(food_nutrients)
    portions = extract_portions(detail) if detail else []
    kcal_estimated = False
    missing = []
    if per["kcal"] is None:
        estimated = estimate_kcal(per)
        if estimated is not None:
            per["kcal"], kcal_estimated = estimated, True
        else:
            missing.append("kcal")
    missing.extend(field for field in REQUIRED if per[field] is None)
    return {"fdcId": fdc_id, "description": description, "source": data_type,
            "per_100g": per, "portions": portions,
            "default_portion": portions[0] if portions else None,
            "missing": missing, "kcal_estimated": kcal_estimated}


def meal_text(name, grams):
    """The 'Dietary intake' string the glucose forecaster reads: '<food> <grams> g'.

    The forecaster keys off the food name (category regex) and the gram amount, so this is
    how a photo/typed food reaches the model unchanged. USDA macros are shown separately.
    """
    return f"{name.strip()} {float(grams):g} g"
