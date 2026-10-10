"""Transparent text-to-numeric food/treatment features; no inferred nutrition."""
import re

import pandas as pd

FOODS = {
    "starch": r"rice|bread|noodle|bun|cereal|oat|corn|potato|yam|porridge|dumpling|pasta|millet",
    "vegetable": r"vegetable|spinach|cabbage|lettuce|broccoli|cucumber|tomato|carrot|lotus|pickle",
    "fruit": r"apple|banana|orange|pear|fruit|grape|melon|berry|peach|date|kiwi",
    "protein": r"egg|beef|pork|fish|chicken|meat|tofu|protein|shrimp|duck|bean",
    "dairy": r"milk|yogurt|yoghurt|cheese",
    "nuts": r"nut|almond|seed|peanut",
    "sweet": r"sugar|cake|biscuit|cookie|candy|juice|honey",
}
DRUGS = ["metformin", "acarbose", "voglibose", "gliclazide", "dapagliflozin",
         "liraglutide", "pioglitazone", "sitagliptin", "canagliflozin",
         "repaglinide", "glimepiride", "linagliptin", "empagliflozin", "gliquidone"]
INSULINS = ["aspart", "glargine", "degludec", "detemir", "glulisine",
            "novolin", "humulin", "gansulin", "scilin"]


def normalized(series):
    return (series.fillna("").str.normalize("NFKC").str.lower()
            .str.replace("�", "", regex=False).str.replace(r"\s+", " ", regex=True).str.strip())


def enrich(data):
    data = data.copy()
    food = normalized(data["Dietary intake"])
    is_food = data.food_logged_now.eq(1)
    data["food_normalized"] = food.where(is_food, data["Dietary intake"])
    food_features = []
    for category, pattern in FOODS.items():
        name = f"food_{category}"
        data[name] = (is_food & food.str.contains(pattern, regex=True)).astype(int)
        food_features.append(name)
    data["food_other"] = (is_food & data[food_features].sum(axis=1).eq(0)).astype(int)
    # Grams of described food, NOT grams of carbohydrates. Keep ml separate.
    for unit, suffix in [("g", "grams"), ("ml", "ml")]:
        amounts = food.str.findall(rf"(\d+(?:\.\d+)?)\s*{unit}\b")
        data[f"food_amount_{suffix}"] = amounts.map(lambda v: sum(map(float, v)) if v else 0.0)
    data["food_quantity_unparsed"] = (is_food & data.food_amount_grams.eq(0) & data.food_amount_ml.eq(0)).astype(int)

    sc = normalized(data["Insulin dose - s.c."]).str.replace("glarigine", "glargine", regex=False)
    data["insulin_sc_normalized"] = sc
    doses = sc.str.findall(r"(\d+(?:\.\d+)?)\s*iu\b")
    data["insulin_sc_iu"] = doses.map(lambda v: sum(map(float, v)) if v else 0.0)
    data["insulin_sc_logged"] = sc.ne("").astype(int)
    data["insulin_sc_unparsed"] = (sc.ne("") & doses.str.len().eq(0)).astype(int)
    for name in INSULINS:
        data[f"insulin_{name}"] = sc.str.contains(name, regex=False).astype(int)
    data["insulin_premixed"] = sc.str.contains(r"70/30|30r|40r|m30", regex=True).astype(int)

    med = normalized(data["Non-insulin hypoglycemic agents"])
    data["medication_normalized"] = med
    data["medication_logged"] = med.ne("").astype(int)
    matched = pd.Series(False, index=data.index)
    for drug in DRUGS:
        present = med.str.contains(drug, regex=False)
        matched |= present
        data[f"med_{drug}"] = present.astype(int)
        extracted = med.str.extract(rf"\b{drug}\s+(\d+(?:\.\d+)?)\s*(mg|g)\b")
        dose = pd.to_numeric(extracted[0], errors="coerce")
        dose = dose * extracted[1].map({"mg": 1, "g": 1000})
        # Preserve ambiguous source units in raw text, but do not feed them as doses.
        ambiguous = present & (extracted[0].isna() | (extracted[1].eq("g") & (drug != "metformin")))
        data[f"med_{drug}_dose_unparsed"] = ambiguous.astype(int)
        data[f"med_{drug}_mg"] = dose.mask(ambiguous).where(present, 0)
    data["medication_unrecognized"] = (med.ne("") & ~matched).astype(int)
    for source, name in [("CSII - bolus insulin (Novolin R, IU)", "csii_bolus_iu"),
                         ("CSII - basal insulin (Novolin R, IU / H)", "csii_basal_iu_h")]:
        text = normalized(data[source])
        numeric = pd.to_numeric(text, errors="coerce")
        data[name] = numeric  # Blank pump rates are unknown; never forward-fill them.
        data[name + "_logged"] = text.ne("").astype(int)
        data[name + "_unparsed"] = (text.ne("") & numeric.isna()).astype(int)
    return data


def audit(data, output):
    for column, filename in [("Dietary intake", "food_mapping.csv"),
                             ("Insulin dose - s.c.", "insulin_mapping.csv"),
                             ("Non-insulin hypoglycemic agents", "medication_mapping.csv")]:
        prefix = "food_" if filename.startswith("food") else "insulin_" if filename.startswith("insulin") else "med_"
        selected = [c for c in data if c.startswith(prefix)]
        data[[column, *selected]].drop_duplicates().to_csv(output / filename, index=False)
    flags = [c for c in data if c.endswith("unparsed") or c.endswith("unrecognized")]
    data.loc[data[flags].sum(axis=1).gt(0),
             ["Patient", "Recording", "Date", "Dietary intake", "Insulin dose - s.c.",
              "Non-insulin hypoglycemic agents", "CSII - bolus insulin (Novolin R, IU)",
              "CSII - basal insulin (Novolin R, IU / H)", *flags]].to_csv(output / "parsing_review.csv", index=False)
