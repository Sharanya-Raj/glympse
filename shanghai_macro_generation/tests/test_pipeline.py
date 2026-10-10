import pandas as pd
import pytest

from src.llm_estimator import _EstimateBatch, _EstimateOut
from src.parse_meals import ParsedMealBatch
from src.pipeline import PipelineOptions, run_pipeline

from .conftest import FakeOpenAI, FakeProvider
from .test_parser import MEALS, good_handler


def handler(payload, response_format):
    if response_format is ParsedMealBatch:
        return good_handler(payload, response_format)
    return _EstimateBatch(estimates=[
        _EstimateOut(food_id=p["food_id"], food_name=p["food_name"], calories_per_100g=100, carbs_per_100g=10,
                     protein_per_100g=5, fat_per_100g=4.4, confidence=0.6, reasoning_summary="estimate")
        for p in payload
    ])


def test_end_to_end_with_mocks(settings, tmp_path):
    input_csv = tmp_path / "meals.csv"
    pd.DataFrame({"meal_id": list(MEALS), "meal_text": list(MEALS.values()), "patient_id": [7, 8]}).to_csv(
        input_csv, index=False
    )
    out = tmp_path / "out" / "meals_with_macros.csv"
    provider = FakeProvider({"orange": (47, 11.8, 0.9, 0.1), "rice": (130, 28, 2.7, 0.3)})
    fake = FakeOpenAI(handler)

    summary = run_pipeline(PipelineOptions(input_path=input_csv, output_path=out), settings,
                           llm_client=fake, providers=[provider])

    meals = pd.read_csv(out, dtype={"meal_id": str}).set_index("meal_id")
    foods = pd.read_csv(out.with_name("food_breakdown.csv"), dtype={"meal_id": str})
    assert summary.processed_meals == 2 and summary.failed == 0
    assert meals.loc["2", "calories"] == pytest.approx(130 * 1.5 + 100)
    assert meals.loc["2", "num_database_matches"] == 1 and meals.loc["2", "num_llm_estimates"] == 1
    assert meals.loc["1", "patient_id"] == 7
    orange = foods[foods.normalized_food_name == "orange"].iloc[0]
    assert orange.calories == pytest.approx(47 * 0.6)
    assert orange.nutrition_source == "database_direct"
    assert out.with_name("run_metadata.json").exists()

    # Rerun: everything comes from cache, no API or database calls.
    calls_before = len(fake.calls)
    provider2 = FakeProvider({})
    summary2 = run_pipeline(PipelineOptions(input_path=input_csv, output_path=out), settings,
                            llm_client=FakeOpenAI(handler), providers=[provider2])
    assert provider2.queries == [] and summary2.llm_usage["requests"] == 0
    assert len(fake.calls) == calls_before
    assert summary2.meals_from_cache == 2


def test_custom_column_names_and_limit(settings, tmp_path):
    input_csv = tmp_path / "meals.csv"
    pd.DataFrame({"id": ["a", "b"], "diet": list(MEALS.values())}).to_csv(input_csv, index=False)
    out = tmp_path / "meals_with_macros.csv"
    summary = run_pipeline(
        PipelineOptions(input_path=input_csv, output_path=out, id_column="id", text_column="diet", limit=1,
                        skip_llm_fallback=True),
        settings, llm_client=FakeOpenAI(handler), providers=[FakeProvider({"orange": (47, 11.8, 0.9, 0.1)})],
    )
    assert summary.processed_meals == 1
    assert summary.nutrition_sources.get("unresolved") == 2  # ginger juice, walnuts and dates
