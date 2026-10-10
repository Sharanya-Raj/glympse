"""Central configuration: paths, column names, model settings, prices, thresholds.

Values can be overridden through environment variables (loaded from `.env`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

# USD per 1M tokens (paid tier). Update when provider pricing changes; MODEL_INPUT_COST_PER_MILLION /
# MODEL_OUTPUT_COST_PER_MILLION in the environment take precedence over this table.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
}

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODELS = {PROVIDER_OPENAI: "gpt-4.1-mini", PROVIDER_GEMINI: "gemini-2.5-flash"}
GOOGLE_KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_AI", "GOOGLE_AI_API_KEY")
USDA_DEMO_KEY = "DEMO_KEY"


def load_env() -> None:
    """Load the project `.env`, then the repo-root `.env` (never overriding values already set)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_float(name: str, default: float | None) -> float | None:
    value = _env_str(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, value)
        return default


def _env_int(name: str, default: int) -> int:
    value = _env_float(name, None)
    return default if value is None else int(value)


def _google_key() -> str | None:
    for name in GOOGLE_KEY_VARS:
        value = _env_str(name)
        if value:
            return value
    return None


@dataclass
class Settings:
    # LLM (OpenAI, or Gemini through Google's OpenAI-compatible endpoint)
    llm_provider: str = PROVIDER_OPENAI
    llm_api_key: str | None = None
    llm_model: str = DEFAULT_MODELS[PROVIDER_OPENAI]
    llm_base_url: str | None = None
    llm_temperature: float | None = 0.0
    llm_reasoning_effort: str | None = None
    llm_timeout_s: float = 180.0
    llm_requests_per_minute: float | None = None
    # Stop calling the LLM after this many consecutive failed batches (e.g. daily quota exhausted).
    llm_max_consecutive_failures: int = 3

    usda_api_key: str = USDA_DEMO_KEY

    # Input columns
    id_column: str = "meal_id"
    text_column: str = "meal_text"

    # Batching / retries
    batch_size: int = 25
    fallback_batch_size: int = 25
    max_retries: int = 6
    backoff_base_s: float = 2.0
    backoff_max_s: float = 90.0
    nutrition_checkpoint_every: int = 5

    # Save states: rotating snapshots of every cache, taken every N checkpoint writes
    # (at most once per snapshot_min_interval_s), plus one at the end of every stage
    snapshot_every: int = 5
    snapshot_keep: int = 30
    snapshot_min_interval_s: float = 120.0
    parse_autosave_every: int = 200

    # Pricing (USD per 1M tokens); None means unknown -> cost not reported
    model_input_cost_per_million: float | None = None
    model_output_cost_per_million: float | None = None

    # Prompt versions are stored with every cached LLM result for reproducibility
    parser_prompt_version: str = "parser-v1"
    estimator_prompt_version: str = "estimator-v1"

    # USDA FoodData Central
    usda_data_types: tuple[str, ...] = ("Foundation", "SR Legacy", "Survey (FNDDS)")
    usda_page_size: int = 50
    usda_requests_per_hour: int = 900
    usda_timeout_s: float = 30.0

    # Database match thresholds (on the 0-1 match score)
    match_direct_threshold: float = 0.90
    match_approximate_threshold: float = 0.80

    # LLM estimate confidence caps
    llm_max_confidence: float = 0.69
    llm_mixed_dish_confidence_range: tuple[float, float] = (0.30, 0.49)
    llm_min_confidence: float = 0.10

    # Meal confidence
    confidence_strategy: str = "weight_weighted"

    # Paths
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "cache")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    @property
    def parsed_cache_path(self) -> Path:
        return self.cache_dir / "parsed_meals.json"

    @property
    def nutrition_cache_path(self) -> Path:
        return self.cache_dir / "nutrition_cache.json"

    @property
    def snapshot_dir(self) -> Path:
        return self.cache_dir / "snapshots"

    @property
    def journal_path(self) -> Path:
        return self.cache_dir / "progress.jsonl"

    @property
    def overrides_path(self) -> Path:
        return self.data_dir / "manual_nutrition_overrides.csv"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def using_usda_demo_key(self) -> bool:
        return self.usda_api_key == USDA_DEMO_KEY

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        openai_key = _env_str("OPENAI_API_KEY")
        google_key = _google_key()
        provider = (_env_str("LLM_PROVIDER") or "").lower()
        if not provider:
            provider = PROVIDER_OPENAI if openai_key or not google_key else PROVIDER_GEMINI
        if provider not in (PROVIDER_OPENAI, PROVIDER_GEMINI):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'gemini', got {provider!r}")

        if provider == PROVIDER_GEMINI:
            api_key, base_url, rpm_default = google_key, GEMINI_OPENAI_BASE_URL, 8.0
            model = _env_str("LLM_MODEL") or _env_str("GEMINI_MODEL") or DEFAULT_MODELS[provider]
        else:
            api_key, base_url, rpm_default = openai_key, _env_str("OPENAI_BASE_URL"), None
            model = _env_str("LLM_MODEL") or _env_str("OPENAI_MODEL") or DEFAULT_MODELS[provider]

        default_in, default_out = MODEL_PRICES.get(model, (None, None))
        usda_key = _env_str("USDA_API_KEY", USDA_DEMO_KEY)
        settings = cls(
            llm_provider=provider,
            llm_api_key=api_key,
            llm_model=model,
            llm_base_url=base_url,
            llm_temperature=_env_float("LLM_TEMPERATURE", _env_float("OPENAI_TEMPERATURE", 0.0)),
            llm_reasoning_effort=_env_str("LLM_REASONING_EFFORT"),
            llm_requests_per_minute=_env_float("LLM_REQUESTS_PER_MINUTE", rpm_default),
            usda_api_key=usda_key,
            batch_size=_env_int("BATCH_SIZE", 25),
            fallback_batch_size=_env_int("FALLBACK_BATCH_SIZE", 25),
            snapshot_every=_env_int("SNAPSHOT_EVERY", 5),
            snapshot_keep=_env_int("SNAPSHOT_KEEP", 30),
            snapshot_min_interval_s=_env_float("SNAPSHOT_MIN_INTERVAL_S", 120.0),
            model_input_cost_per_million=_env_float("MODEL_INPUT_COST_PER_MILLION", default_in),
            model_output_cost_per_million=_env_float("MODEL_OUTPUT_COST_PER_MILLION", default_out),
            usda_requests_per_hour=_env_int(
                "USDA_REQUESTS_PER_HOUR", 30 if usda_key == USDA_DEMO_KEY else 900
            ),
        )
        if settings.model_input_cost_per_million is None:
            logger.warning(
                "No price configured for model %s; set MODEL_INPUT_COST_PER_MILLION and "
                "MODEL_OUTPUT_COST_PER_MILLION to get cost estimates.",
                model,
            )
        return settings
