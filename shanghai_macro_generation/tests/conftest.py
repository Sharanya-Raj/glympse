from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.models import SOURCE_DB_DIRECT, SOURCE_DB_REJECTED, NutritionResult


@pytest.fixture
def settings(tmp_path) -> Settings:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return Settings(
        llm_api_key="test-key",
        llm_model="test-model",
        usda_api_key="test",
        cache_dir=tmp_path / "cache",
        data_dir=data_dir,
        log_dir=tmp_path / "logs",
        max_retries=2,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        model_input_cost_per_million=1.0,
        model_output_cost_per_million=2.0,
    )


def completion(parsed, prompt_tokens: int = 100, completion_tokens: int = 50):
    message = SimpleNamespace(parsed=parsed, refusal=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeOpenAI:
    """Mimics `client.chat.completions.parse`; `handler(payload, response_format)` builds the parsed object."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, *, model, messages, response_format, **kwargs):
        user = messages[-1]["content"]
        payload = json.loads(user[user.index("[") :])
        self.calls.append({"payload": payload, "response_format": response_format})
        result = self.handler(payload, response_format)
        if isinstance(result, Exception):
            raise result
        return completion(result)


class FakeProvider:
    """Nutrition database stub: exact-name table; everything else is a rejected candidate."""

    name = "fake_db"

    def __init__(self, table: dict[str, tuple[float, float, float, float]]):
        self.table = table
        self.queries: list[str] = []

    def search(self, food_name: str) -> NutritionResult | None:
        self.queries.append(food_name)
        if food_name in self.table:
            kcal, carbs, protein, fat = self.table[food_name]
            return NutritionResult(
                normalized_name=food_name, calories_per_100g=kcal, carbs_per_100g=carbs,
                protein_per_100g=protein, fat_per_100g=fat, source=SOURCE_DB_DIRECT, confidence=0.98,
                matched_food_name=f"{food_name}, raw", match_score=1.0,
            )
        return NutritionResult(
            normalized_name=food_name, calories_per_100g=100, carbs_per_100g=10, protein_per_100g=5,
            fat_per_100g=4, source=SOURCE_DB_REJECTED, confidence=0.0,
            matched_food_name="Something else", fdc_id=42, match_score=0.2,
        )
