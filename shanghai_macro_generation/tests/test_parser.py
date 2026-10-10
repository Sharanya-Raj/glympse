import openai
import pytest

from src.cache import JsonCache
from src.llm_client import LLMClient, UsageTracker
from src.llm_estimator import LLMNutritionEstimator, _EstimateBatch, _EstimateOut
from src.parse_meals import PARSER_RULE, MealParser, ParsedMealBatch, _FoodOut, _MealOut, rule_parse_meal

from .conftest import FakeOpenAI

MEALS = {
    "1": "Ginger juice 100 ml Walnuts and dates 15 g Orange 60 g",
    "2": "Rice 150 g; Steamed egg 100 g",
}

GOOD = {
    MEALS["1"]: [
        _FoodOut(original_text="Ginger juice 100 ml", name="ginger juice", amount=100, unit="ml", preparation=None),
        _FoodOut(original_text="Walnuts and dates 15 g", name="walnuts and dates", amount=15, unit="g", preparation=None),
        _FoodOut(original_text="Orange 60 g", name="orange", amount=60, unit="g", preparation=None),
    ],
    MEALS["2"]: [
        _FoodOut(original_text="Rice 150 g", name="rice", amount=150, unit="g", preparation=None),
        _FoodOut(original_text="Steamed egg 100 g", name="steamed egg", amount=100, unit="g", preparation="steamed"),
    ],
}


def good_handler(payload, response_format):
    return ParsedMealBatch(meals=[_MealOut(meal_id=m["meal_id"], foods=GOOD[m["meal_text"]]) for m in payload])


def make_parser(settings, handler, **kwargs):
    fake = FakeOpenAI(handler)
    usage = UsageTracker(1.0, 2.0)
    llm = LLMClient(settings, usage, client=fake)
    cache = JsonCache(settings.parsed_cache_path)
    return MealParser(cache, settings, llm=llm, **kwargs), fake, usage


def test_batch_parse_and_cache(settings):
    parser, fake, usage = make_parser(settings, good_handler)
    out = parser.parse_meals(list(MEALS.items()))
    assert len(fake.calls) == 1  # both meals in one request
    assert [f.name for f in out.parsed["1"].foods] == ["ginger juice", "walnuts and dates", "orange"]
    assert out.parsed["2"].foods[1].preparation == "steamed"
    assert usage.usage.requests == 1 and usage.estimated_cost == pytest.approx((100 * 1 + 50 * 2) / 1e6)

    parser2, fake2, _ = make_parser(settings, good_handler)
    out2 = parser2.parse_meals(list(MEALS.items()))
    assert fake2.calls == [] and out2.cached_meals == 2


def test_duplicate_texts_are_parsed_once(settings):
    parser, fake, _ = make_parser(settings, good_handler)
    out = parser.parse_meals([("a", MEALS["1"]), ("b", MEALS["1"])])
    assert len(fake.calls[0]["payload"]) == 1
    assert set(out.parsed) == {"a", "b"}


def test_batching_respects_batch_size(settings):
    parser, fake, _ = make_parser(settings, good_handler, batch_size=1)
    parser.parse_meals(list(MEALS.items()))
    assert len(fake.calls) == 2


def test_missing_meal_in_batch_is_retried_individually(settings):
    def handler(payload, response_format):
        if len(payload) > 1:  # drop the second meal from the batch response
            return ParsedMealBatch(meals=[_MealOut(meal_id=payload[0]["meal_id"], foods=GOOD[payload[0]["meal_text"]])])
        return good_handler(payload, response_format)

    parser, fake, _ = make_parser(settings, handler)
    out = parser.parse_meals(list(MEALS.items()))
    assert len(fake.calls) == 2 and not out.failures
    assert set(out.parsed) == {"1", "2"}


def test_invented_amount_triggers_retry_then_warning(settings):
    def handler(payload, response_format):
        meals = []
        for m in payload:
            foods = [f.model_copy() for f in GOOD[m["meal_text"]]]
            foods[0].amount = 999  # not present in the text
            meals.append(_MealOut(meal_id=m["meal_id"], foods=foods))
        return ParsedMealBatch(meals=meals)

    parser, fake, _ = make_parser(settings, handler)
    out = parser.parse_meals([("1", MEALS["1"])])
    assert len(fake.calls) == 2
    assert any("999" in w for w in out.parsed["1"].warnings)


def test_failed_batch_is_logged_and_other_batches_continue(settings):
    def handler(payload, response_format):
        if payload[0]["meal_text"] == MEALS["1"]:
            return openai.APIConnectionError(request=None)
        return good_handler(payload, response_format)

    parser, fake, usage = make_parser(settings, handler, batch_size=1)
    out = parser.parse_meals(list(MEALS.items()))
    assert "1" in out.failures and "2" in out.parsed
    assert len(fake.calls) == settings.max_retries + 1  # retries for batch 1, then batch 2
    assert usage.usage.failed_requests == settings.max_retries
    # failures are not cached, so a rerun tries again
    assert JsonCache(settings.parsed_cache_path).get(parser._cache_key(MEALS["1"])) is None


def test_rule_parser_keeps_mixed_dish_whole():
    foods = rule_parse_meal("Lotus nut, red bean, poria coco and orange peel 30 g Dry-fried pomfret 120 g")
    assert [(f.name, f.amount, f.unit) for f in foods] == [
        ("lotus nut, red bean, poria coco and orange peel", 30, "g"),
        ("dry-fried pomfret", 120, "g"),
    ]
    assert foods[1].preparation == "dry-fried"


def test_rule_parser_semicolons_and_missing_amount(settings):
    foods = rule_parse_meal("Milk 250 ml; Egg 50 g; Vegetables")
    assert [(f.name, f.amount, f.unit) for f in foods] == [("milk", 250, "ml"), ("egg", 50, "g"), ("vegetables", None, None)]
    parser = MealParser(JsonCache(settings.parsed_cache_path), settings, mode=PARSER_RULE)
    assert not parser.parse_meals([("1", "Milk 250 ml; Egg 50 g")]).failures


def test_llm_estimator_clamps_confidence_and_flags_implausible(settings):
    def handler(payload, response_format):
        assert response_format is _EstimateBatch
        return _EstimateBatch(estimates=[
            _EstimateOut(food_id="f1", food_name="braised duck wing", calories_per_100g=250, carbs_per_100g=5,
                         protein_per_100g=20, fat_per_100g=17, confidence=0.95, reasoning_summary="typical"),
            _EstimateOut(food_id="f2", food_name="lotus nut, red bean", calories_per_100g=900, carbs_per_100g=60,
                         protein_per_100g=10, fat_per_100g=5, confidence=0.6, reasoning_summary="mix"),
        ])

    llm = LLMClient(settings, UsageTracker(None, None), client=FakeOpenAI(handler))
    out = LLMNutritionEstimator(llm, settings).estimate(["braised duck wing", "lotus nut, red bean"])
    duck, mix = out["braised duck wing"], out["lotus nut, red bean"]
    assert duck.source == "llm_estimate" and duck.confidence == pytest.approx(0.69)
    assert "mixed_dish" in mix.flags and "atwater_mismatch" in mix.flags
    assert mix.confidence < 0.49
