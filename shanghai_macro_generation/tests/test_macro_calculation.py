import pytest

from src.macro_calculator import (
    aggregate_meal,
    calorie_weighted_confidence,
    scale_macros,
    weight_weighted_confidence,
)
from src.models import (
    SOURCE_DB_DIRECT,
    SOURCE_LLM,
    SOURCE_UNRESOLVED,
    FoodResult,
    NutritionResult,
)


def nutrition(kcal=100.0, carbs=20.0, protein=5.0, fat=2.0, source=SOURCE_DB_DIRECT, conf=1.0):
    return NutritionResult(
        normalized_name="x", calories_per_100g=kcal, carbs_per_100g=carbs, protein_per_100g=protein,
        fat_per_100g=fat, source=source, confidence=conf,
    )


def food(grams, kcal, carbs=0.0, protein=0.0, fat=0.0, source=SOURCE_DB_DIRECT, conf=1.0, unit_conf=1.0):
    resolved = source != SOURCE_UNRESOLVED
    return FoodResult(
        meal_id="m", original_food_text="t", original_food_name="t", normalized_food_name="t",
        amount=grams, unit="g", estimated_grams=grams,
        calories=kcal if resolved else None, carbs_g=carbs if resolved else None,
        protein_g=protein if resolved else None, fat_g=fat if resolved else None,
        nutrition_source=source, nutrition_confidence=conf if resolved else 0.0,
        unit_conversion_confidence=unit_conf,
    )


def test_100g_gives_per_100g_values():
    assert scale_macros(nutrition(carbs=20), 100)["carbs_g"] == pytest.approx(20)


def test_50g_halves_values():
    out = scale_macros(nutrition(kcal=100, carbs=20, protein=5, fat=2), 50)
    assert out == pytest.approx({"calories": 50, "carbs_g": 10, "protein_g": 2.5, "fat_g": 1})


def test_missing_grams_or_nutrition_gives_none():
    assert scale_macros(nutrition(), None)["calories"] is None
    assert scale_macros(None, 100)["calories"] is None


def test_meal_sums_foods():
    foods = [food(100, 130, carbs=28, protein=2.7, fat=0.3), food(60, 30, carbs=7, protein=0.5, fat=0.1)]
    meal = aggregate_meal("1", "Rice 100 g Orange 60 g", foods)
    assert meal.calories == pytest.approx(160)
    assert meal.carbs_g == pytest.approx(35)
    assert meal.protein_g == pytest.approx(3.2)
    assert meal.fat_g == pytest.approx(0.4)
    assert meal.num_food_items == 2 and meal.num_database_matches == 2 and meal.status == "ok"


def test_weight_weighted_confidence():
    foods = [food(100, 100, conf=1.0), food(300, 100, source=SOURCE_LLM, conf=0.5)]
    assert weight_weighted_confidence(foods) == pytest.approx((1.0 * 100 + 0.5 * 300) / 400)


def test_unit_conversion_confidence_is_included():
    foods = [food(100, 100, conf=1.0, unit_conf=0.5)]
    assert weight_weighted_confidence(foods) == pytest.approx(0.5)


def test_calorie_weighted_confidence():
    foods = [food(100, 300, conf=1.0), food(100, 100, source=SOURCE_LLM, conf=0.6)]
    assert calorie_weighted_confidence(foods) == pytest.approx((300 * 1.0 + 100 * 0.6) / 400)


def test_unresolved_items_reduce_confidence_and_mark_partial():
    foods = [food(100, 100, conf=1.0), food(100, 0, source=SOURCE_UNRESOLVED)]
    meal = aggregate_meal("1", "t", foods)
    assert meal.status == "partial"
    assert meal.num_unresolved_items == 1
    assert meal.confidence == pytest.approx(0.5)
    assert meal.calories == pytest.approx(100)


def test_parse_failure_and_empty_meals():
    assert aggregate_meal("1", "t", [], parse_failed=True).status == "parse_failed"
    assert aggregate_meal("1", "", []).status == "empty"
