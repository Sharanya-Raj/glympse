"""Scale per-100 g nutrition by served grams, sum foods into meals, score meal confidence."""

from __future__ import annotations

from typing import Callable

from .models import (
    DATABASE_SOURCES,
    MEAL_STATUS_EMPTY,
    MEAL_STATUS_NO_NUTRITION,
    MEAL_STATUS_OK,
    MEAL_STATUS_PARSE_FAILED,
    MEAL_STATUS_PARTIAL,
    SOURCE_DB_APPROX,
    SOURCE_DB_DIRECT,
    SOURCE_LLM,
    SOURCE_MANUAL,
    FoodResult,
    MealResult,
    NutritionResult,
)

MACRO_FIELDS = {
    "calories": "calories_per_100g",
    "carbs_g": "carbs_per_100g",
    "protein_g": "protein_per_100g",
    "fat_g": "fat_per_100g",
}


def scale_macros(nutrition: NutritionResult | None, grams: float | None) -> dict[str, float | None]:
    """Macros for `grams` of a food: value_per_100g * grams / 100. None where not computable."""
    out: dict[str, float | None] = {}
    for field, per100_field in MACRO_FIELDS.items():
        per100 = getattr(nutrition, per100_field, None) if nutrition is not None else None
        out[field] = None if per100 is None or grams is None else per100 * grams / 100.0
    return out


# ---------------------------------------------------------------------------
# Meal confidence strategies: (foods) -> confidence in [0, 1] or None
# ---------------------------------------------------------------------------

ConfidenceStrategy = Callable[[list[FoodResult]], "float | None"]


def _weighted_confidence(foods: list[FoodResult], weight: Callable[[FoodResult], float]) -> float | None:
    if not foods:
        return None
    resolved = [f for f in foods if f.is_resolved]
    if not resolved:
        return 0.0
    scores = [f.nutrition_confidence * f.unit_conversion_confidence for f in resolved]
    weights = [max(0.0, weight(f)) for f in resolved]
    total = sum(weights)
    base = sum(s * w for s, w in zip(scores, weights)) / total if total > 0 else sum(scores) / len(scores)
    # Unresolved items mean part of the meal is missing from the totals.
    return base * len(resolved) / len(foods)


def weight_weighted_confidence(foods: list[FoodResult]) -> float | None:
    """sum(nutrition_conf * unit_conf * grams) / sum(grams), times the share of resolved items."""
    return _weighted_confidence(foods, lambda f: f.estimated_grams or 0.0)


def calorie_weighted_confidence(foods: list[FoodResult]) -> float | None:
    """Same as weight_weighted but weighted by each item's calories (energy-dense items matter more)."""
    return _weighted_confidence(foods, lambda f: f.calories or 0.0)


def min_item_confidence(foods: list[FoodResult]) -> float | None:
    """Conservative: the weakest item determines the meal confidence."""
    if not foods:
        return None
    return min(
        (f.nutrition_confidence * f.unit_conversion_confidence if f.is_resolved else 0.0) for f in foods
    )


CONFIDENCE_STRATEGIES: dict[str, ConfidenceStrategy] = {
    "weight_weighted": weight_weighted_confidence,
    "calorie_weighted": calorie_weighted_confidence,
    "min_item": min_item_confidence,
}


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def aggregate_meal(
    meal_id: str,
    meal_text: str,
    foods: list[FoodResult],
    strategy: str = "weight_weighted",
    parse_failed: bool = False,
    parse_warnings: list[str] | None = None,
) -> MealResult:
    if parse_failed:
        return MealResult(meal_id=meal_id, meal_text=meal_text, status=MEAL_STATUS_PARSE_FAILED, confidence=0.0,
                          parse_warnings="; ".join(parse_warnings or []) or None)
    if not foods:
        return MealResult(meal_id=meal_id, meal_text=meal_text, status=MEAL_STATUS_EMPTY)

    resolved = [f for f in foods if f.is_resolved]
    totals = {k: (sum(getattr(f, k) for f in resolved) if resolved else None) for k in MACRO_FIELDS}
    sources = [f.nutrition_source for f in foods]
    if not resolved:
        status = MEAL_STATUS_NO_NUTRITION
    elif len(resolved) < len(foods):
        status = MEAL_STATUS_PARTIAL
    else:
        status = MEAL_STATUS_OK

    confidence = CONFIDENCE_STRATEGIES[strategy](foods)
    return MealResult(
        meal_id=meal_id,
        meal_text=meal_text,
        calories=_round(totals["calories"], 1),
        carbs_g=_round(totals["carbs_g"], 2),
        protein_g=_round(totals["protein_g"], 2),
        fat_g=_round(totals["fat_g"], 2),
        confidence=_round(confidence, 3),
        num_food_items=len(foods),
        num_database_matches=sum(s in DATABASE_SOURCES for s in sources),
        num_llm_estimates=sources.count(SOURCE_LLM),
        num_db_direct=sources.count(SOURCE_DB_DIRECT),
        num_db_approximate=sources.count(SOURCE_DB_APPROX),
        num_manual_overrides=sources.count(SOURCE_MANUAL),
        num_unresolved_items=len(foods) - len(resolved),
        total_grams=_round(sum(f.estimated_grams for f in resolved) if resolved else None, 1),
        status=status,
        parse_warnings="; ".join(parse_warnings or []) or None,
    )
