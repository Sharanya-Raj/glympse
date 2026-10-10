"""Pydantic data models shared across the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Nutrition sources. Every per-food estimate carries exactly one of these.
SOURCE_MANUAL = "manual_override"
SOURCE_DB_DIRECT = "database_direct"
SOURCE_DB_APPROX = "database_approximate"
SOURCE_DB_REJECTED = "database_rejected"
SOURCE_DB_MISS = "db_miss"
SOURCE_LLM = "llm_estimate"
SOURCE_UNRESOLVED = "unresolved"

DATABASE_SOURCES = frozenset({SOURCE_DB_DIRECT, SOURCE_DB_APPROX})
# Sources whose values can be used to compute macros and reused from the cache.
FINAL_SOURCES = frozenset({SOURCE_MANUAL, SOURCE_DB_DIRECT, SOURCE_DB_APPROX, SOURCE_LLM})

MEAL_STATUS_OK = "ok"
MEAL_STATUS_PARTIAL = "partial"
MEAL_STATUS_NO_NUTRITION = "no_nutrition"
MEAL_STATUS_PARSE_FAILED = "parse_failed"
MEAL_STATUS_EMPTY = "empty"
SUCCESS_STATUSES = frozenset({MEAL_STATUS_OK, MEAL_STATUS_PARTIAL})


class FoodItem(BaseModel):
    original_text: str
    name: str
    amount: float | None
    unit: str | None
    preparation: str | None = None


class ParsedMeal(BaseModel):
    meal_id: str | int
    foods: list[FoodItem]
    parser: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    warnings: list[str] = Field(default_factory=list)


class NutritionResult(BaseModel):
    normalized_name: str

    calories_per_100g: float | None
    carbs_per_100g: float | None
    protein_per_100g: float | None
    fat_per_100g: float | None

    source: str
    confidence: float

    matched_food_name: str | None = None
    fdc_id: int | None = None
    match_score: float | None = None
    query: str | None = None

    # Best database candidate when the database match was rejected (kept for traceability).
    db_candidate_name: str | None = None
    db_candidate_fdc_id: int | None = None
    db_candidate_score: float | None = None

    reasoning_summary: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    notes: str | None = None
    flags: list[str] = Field(default_factory=list)
    created_at: str | None = None

    @property
    def has_macros(self) -> bool:
        return all(
            v is not None
            for v in (self.calories_per_100g, self.carbs_per_100g, self.protein_per_100g, self.fat_per_100g)
        )


class FoodResult(BaseModel):
    """One food item of one meal with its scaled macros and full provenance."""

    meal_id: str
    original_food_text: str
    original_food_name: str
    normalized_food_name: str
    preparation: str | None = None
    amount: float | None
    unit: str | None
    estimated_grams: float | None
    calories: float | None = None
    carbs_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    nutrition_source: str
    nutrition_confidence: float
    unit_conversion_confidence: float
    matched_food_name: str | None = None
    fdc_id: int | None = None
    match_score: float | None = None
    reasoning_summary: str | None = None
    nutrition_flags: str | None = None
    parser: str | None = None
    parser_model: str | None = None
    estimator_model: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.estimated_grams is not None and self.calories is not None


class MealResult(BaseModel):
    meal_id: str
    meal_text: str
    calories: float | None = None
    carbs_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    confidence: float | None = None
    num_food_items: int = 0
    num_database_matches: int = 0
    num_llm_estimates: int = 0
    num_db_direct: int = 0
    num_db_approximate: int = 0
    num_manual_overrides: int = 0
    num_unresolved_items: int = 0
    total_grams: float | None = None
    status: str = MEAL_STATUS_OK
    parse_warnings: str | None = None


class LLMUsage(BaseModel):
    requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    by_purpose: dict[str, dict[str, int]] = Field(default_factory=dict)
