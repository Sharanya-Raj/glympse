"""LLM fallback: per-100 g macro estimates for foods the database could not match confidently."""

from __future__ import annotations

import json
import logging
from typing import Callable

from pydantic import BaseModel

from .cache import utc_now_iso
from .config import Settings
from .llm_client import FailureBreaker, LLMClient, LLMUnavailableError
from .models import SOURCE_LLM, NutritionResult
from .normalize_food import is_mixed_dish

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You estimate typical nutrient content PER 100 g of a food as served, for foods \
recorded in a clinical CGM study in Shanghai (Chinese hospital and home cooking, names translated \
to English).

For each food return calories (kcal), carbohydrates (g), protein (g) and fat (g) per 100 g of the \
edible food as eaten (cooked weight for cooked dishes, including typical cooking oil and sauce for \
that dish). Values must be physically plausible: carbs + protein + fat <= 100 g, and calories close \
to 4*carbs + 4*protein + 9*fat (alcohol adds 7 kcal/g).

confidence (0-1) must follow these bands:
- 0.50-0.69: a recognizable single prepared food or dish with a typical recipe
- 0.30-0.49: a mixed dish or list of ingredients with unknown proportions, or an ambiguous name
- below 0.30: you are guessing what the food is

reasoning_summary: one short sentence naming the reference food or assumption used \
(e.g. "Typical Chinese braised duck wing with soy-sugar sauce, skin on."). No step-by-step reasoning.
Return one estimate per input food_id."""


class _EstimateOut(BaseModel):
    food_id: str
    food_name: str
    calories_per_100g: float
    carbs_per_100g: float
    protein_per_100g: float
    fat_per_100g: float
    confidence: float
    reasoning_summary: str


class _EstimateBatch(BaseModel):
    estimates: list[_EstimateOut]


def postprocess_estimate(name: str, est: _EstimateOut, settings: Settings, model: str | None) -> NutritionResult | None:
    """Clamp confidence to the LLM bands and run plausibility checks. Returns None if unusable."""
    values = (est.calories_per_100g, est.carbs_per_100g, est.protein_per_100g, est.fat_per_100g)
    if any(v < 0 for v in values):
        logger.warning("Discarding LLM estimate for %r: negative values %s", name, values)
        return None

    flags: list[str] = []
    confidence = min(max(est.confidence, settings.llm_min_confidence), settings.llm_max_confidence)
    if is_mixed_dish(name):
        lo, hi = settings.llm_mixed_dish_confidence_range
        confidence = min(max(confidence, lo), hi)
        flags.append("mixed_dish")

    macro_mass = est.carbs_per_100g + est.protein_per_100g + est.fat_per_100g
    if macro_mass > 100:
        flags.append("macro_sum_exceeds_100g")
        confidence *= 0.7
    atwater = 4 * est.carbs_per_100g + 4 * est.protein_per_100g + 9 * est.fat_per_100g
    if est.calories_per_100g > 0 and abs(atwater - est.calories_per_100g) / est.calories_per_100g > 0.25:
        flags.append("atwater_mismatch")
        confidence *= 0.8

    return NutritionResult(
        normalized_name=name,
        calories_per_100g=est.calories_per_100g,
        carbs_per_100g=est.carbs_per_100g,
        protein_per_100g=est.protein_per_100g,
        fat_per_100g=est.fat_per_100g,
        source=SOURCE_LLM,
        confidence=round(max(confidence, settings.llm_min_confidence), 3),
        reasoning_summary=est.reasoning_summary.strip(),
        model=model,
        prompt_version=settings.estimator_prompt_version,
        flags=flags,
        created_at=utc_now_iso(),
    )


class LLMNutritionEstimator:
    def __init__(self, llm: LLMClient, settings: Settings, batch_size: int | None = None) -> None:
        self.llm = llm
        self.settings = settings
        self.model = llm.model
        self.batch_size = max(1, batch_size or settings.fallback_batch_size)

    def _request(self, names: list[str]) -> dict[str, NutritionResult]:
        local = {f"f{i + 1}": name for i, name in enumerate(names)}
        payload = [{"food_id": fid, "food_name": name} for fid, name in local.items()]
        batch = self.llm.parse(
            system=SYSTEM_PROMPT,
            user="Estimate per-100 g nutrition for these foods:\n" + json.dumps(payload, ensure_ascii=False),
            response_format=_EstimateBatch,
            purpose="estimate_nutrition",
        )
        out: dict[str, NutritionResult] = {}
        for est in batch.estimates:
            name = local.get(est.food_id)
            if name is None:
                continue
            result = postprocess_estimate(name, est, self.settings, self.model)
            if result is not None:
                out[name] = result
        return out

    def estimate(
        self,
        food_names: list[str],
        on_batch: Callable[[dict[str, NutritionResult]], None] | None = None,
    ) -> dict[str, NutritionResult]:
        """Estimate in batches. `on_batch` receives each batch's results as soon as they arrive,
        so callers can checkpoint them. Stops early (LLMUnavailableError is logged, not raised)
        after too many consecutive failed batches."""
        results: dict[str, NutritionResult] = {}
        batches = [food_names[i : i + self.batch_size] for i in range(0, len(food_names), self.batch_size)]
        breaker = FailureBreaker(self.settings.llm_max_consecutive_failures)
        for b_idx, batch in enumerate(batches, start=1):
            batch_results: dict[str, NutritionResult] = {}
            try:
                batch_results.update(self._request(batch))
            except Exception as exc:
                logger.error("LLM fallback batch %d/%d failed after retries: %s", b_idx, len(batches), exc)
                try:
                    breaker.failure("fallback")
                except LLMUnavailableError as stop:
                    logger.error("%s", stop)
                    break
                continue
            breaker.success()
            missing = [n for n in batch if n not in batch_results]
            if missing:
                logger.warning("LLM fallback omitted %d foods in batch %d; retrying them once", len(missing), b_idx)
                try:
                    batch_results.update(self._request(missing))
                except Exception as exc:
                    logger.error("LLM fallback retry failed: %s", exc)
            results.update(batch_results)
            if on_batch is not None:
                on_batch(batch_results)
            logger.info("LLM fallback batch %d/%d done (%d estimates)", b_idx, len(batches), len(results))
        return results
