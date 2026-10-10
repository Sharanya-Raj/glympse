"""Unit normalization and conversion of served amounts to grams."""

from __future__ import annotations

import re

UNIT_ALIASES: dict[str, str] = {
    "g": "g", "gr": "g", "gm": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "cc": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "fl oz": "fl oz", "floz": "fl oz", "fluid ounce": "fl oz", "fluid ounces": "fl oz",
}

MASS_TO_GRAMS: dict[str, float] = {"g": 1.0, "kg": 1000.0, "mg": 0.001, "oz": 28.349523125}
VOLUME_TO_ML: dict[str, float] = {"ml": 1.0, "l": 1000.0, "fl oz": 29.5735295625}

# (keywords, density g/ml, conversion confidence); first match wins, so order matters.
DENSITY_RULES: list[tuple[tuple[str, ...], float, float]] = [
    (("oil",), 0.92, 0.85),
    (("honey", "syrup"), 1.40, 0.80),
    (("milk", "yogurt", "yoghurt", "soymilk", "kefir"), 1.03, 0.90),
    (("baijiu", "liquor", "spirit", "vodka", "whisky", "whiskey"), 0.95, 0.80),
    (("wine", "beer"), 1.00, 0.85),
    (("water", "tea", "coffee", "juice", "soda", "drink", "beverage"), 1.00, 0.90),
    (("soup", "broth"), 1.00, 0.85),
    (("congee", "porridge", "gruel"), 1.05, 0.80),
]
UNKNOWN_DENSITY = 1.0
UNKNOWN_DENSITY_CONFIDENCE = 0.5

_WORD_RE = re.compile(r"[a-z]+")


def normalize_unit(unit: str | None) -> str | None:
    """Map unit spellings to a canonical unit ('g', 'kg', 'mg', 'ml', 'l', 'oz', 'fl oz'); None if unknown."""
    if unit is None:
        return None
    key = " ".join(unit.strip().lower().replace(".", "").split())
    return UNIT_ALIASES.get(key)


def density_for(food_name: str) -> tuple[float, float]:
    """Return (density g/ml, confidence) from keyword rules; unknown foods assume water density."""
    words = set(_WORD_RE.findall(food_name.lower()))
    for keywords, density, confidence in DENSITY_RULES:
        if words.intersection(keywords):
            return density, confidence
    return UNKNOWN_DENSITY, UNKNOWN_DENSITY_CONFIDENCE


def convert_to_grams(food_name: str, amount: float | None, unit: str | None) -> tuple[float | None, float]:
    """Convert an amount to grams. Returns (grams or None, conversion confidence 0-1)."""
    if amount is None or amount < 0:
        return None, 0.0
    canonical = normalize_unit(unit)
    if canonical is None:
        return None, 0.0
    if canonical in MASS_TO_GRAMS:
        return amount * MASS_TO_GRAMS[canonical], 1.0
    density, confidence = density_for(food_name)
    return amount * VOLUME_TO_ML[canonical] * density, confidence
