"""Per-100 g nutrition resolution.

Priority: manual override -> cached result -> nutrition database -> LLM fallback.
Database providers implement the `NutritionProvider` protocol so USDA can be swapped out.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pandas as pd

from .cache import NULL_JOURNAL, JsonCache, ProgressJournal, utc_now_iso
from .config import Settings
from .models import (
    DATABASE_SOURCES,
    FINAL_SOURCES,
    SOURCE_DB_APPROX,
    SOURCE_DB_DIRECT,
    SOURCE_DB_MISS,
    SOURCE_DB_REJECTED,
    SOURCE_MANUAL,
    SOURCE_UNRESOLVED,
    NutritionResult,
)
from .normalize_food import is_mixed_dish, normalize_food_name
from .retry import with_retries

logger = logging.getLogger(__name__)


class NutritionProvider(Protocol):
    name: str

    def search(self, food_name: str) -> NutritionResult | None:
        """Best match for `food_name` (per 100 g), or None if nothing usable was found.

        A result whose source is `database_rejected` means a candidate existed but matched poorly.
        """
        ...


class NutritionEstimator(Protocol):
    model: str | None

    def estimate(
        self,
        food_names: list[str],
        on_batch: Callable[[dict[str, NutritionResult]], None] | None = None,
    ) -> dict[str, NutritionResult]:
        ...


# ---------------------------------------------------------------------------
# Food-name match scoring
# ---------------------------------------------------------------------------

STOPWORDS = frozenset({"and", "with", "in", "of", "the", "a", "an", "or", "style", "type", "dish"})

# Descriptor tokens in database names that do not change what the food is.
NEUTRAL_TOKENS = frozenset({
    "raw", "fresh", "cooked", "all", "commercial", "variety", "whole", "regular", "plain",
    "n", "nf", "nfs", "ns", "as", "to", "form", "includes", "include", "usda", "commodity",
    "edible", "portion", "mature", "enriched", "unenriched", "prepared", "from", "recipe",
    "solid", "drained", "without", "added", "salt", "no", "not", "further", "specified",
    "made", "home", "homemade",
})

# Preparation words that materially change macros; missing them in the candidate is penalized.
PREPARATION_TOKENS = frozenset({
    "braised", "fried", "stir", "dry", "deep", "pan", "steamed", "boiled", "roasted", "grilled",
    "stewed", "baked", "sauteed", "smoked", "pickled", "marinated", "sauce", "brown", "sweet",
    "sour", "candied", "salted", "cured", "stuffed", "battered", "breaded",
})

# Foods typically weighed as served (cooked). Raw/dry database forms are penalized for these.
COOKED_STAPLE_TOKENS = frozenset({
    "rice", "noodle", "pasta", "spaghetti", "vermicelli", "macaroni", "millet", "oat", "oatmeal",
    "barley", "quinoa", "buckwheat", "grain", "congee", "porridge", "dumpling", "wonton",
    "beef", "pork", "chicken", "duck", "fish", "mutton", "lamb", "egg", "shrimp", "meat",
})
RAW_FORM_TOKENS = frozenset({"raw", "uncooked"})
# Cutting words describe shape, not composition; they are ignored in queries.
CUT_TOKENS = frozenset({
    "sliced", "slice", "shredded", "shred", "minced", "diced", "dice", "chopped", "cubed", "cube",
    "cut", "piece", "strip", "chunk",
})
DRY_FORM_TOKENS = frozenset({"dry", "dried", "unprepared", "flour", "powder", "dehydrated", "mix"})

DATA_TYPE_PRIORITY = {"Foundation": 0, "SR Legacy": 1, "Survey (FNDDS)": 2}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("oes"):
        return token[:-2]
    if token.endswith(("ches", "shes", "xes", "sses")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class MatchScore:
    score: float
    coverage: float
    precision: float
    num_tokens: int


def match_score(query: str, candidate: str) -> MatchScore:
    """Score how well a database food name matches the query (0-1).

    coverage  = share of query tokens present in the candidate
    precision = share of informative candidate tokens that come from the query
    score     = coverage^2 * (0.5 + 0.5 * precision) - penalties
    """
    q = set(tokenize(query)) - CUT_TOKENS
    c_tokens = tokenize(candidate)
    c = set(c_tokens)
    if not q or not c:
        return MatchScore(0.0, 0.0, 0.0, len(c_tokens))
    overlap = q & c
    coverage = len(overlap) / len(q)
    extra = c - q - NEUTRAL_TOKENS
    precision = len(overlap) / (len(overlap) + len(extra)) if overlap else 0.0
    score = coverage**2 * (0.5 + 0.5 * precision)

    # The first comma segment of a database name is its head food ("Crackers, wonton" is a cracker).
    head = set(tokenize(candidate.split(",")[0])) - NEUTRAL_TOKENS
    if head and not head & q:
        score -= 0.2
    score -= 0.15 * len((q & PREPARATION_TOKENS) - c)
    if not (q & (RAW_FORM_TOKENS | DRY_FORM_TOKENS)):
        if c & DRY_FORM_TOKENS - q:
            score -= 0.3
        if q & COOKED_STAPLE_TOKENS and c & RAW_FORM_TOKENS:
            score -= 0.2
    return MatchScore(max(0.0, min(1.0, score)), coverage, precision, len(c_tokens))


def score_to_source_confidence(score: float, settings: Settings) -> tuple[str, float]:
    direct, approx = settings.match_direct_threshold, settings.match_approximate_threshold
    if score >= direct:
        return SOURCE_DB_DIRECT, round(0.90 + 0.10 * (score - direct) / max(1e-9, 1 - direct), 3)
    if score >= approx:
        return SOURCE_DB_APPROX, round(0.70 + 0.19 * (score - approx) / max(1e-9, direct - approx), 3)
    return SOURCE_DB_REJECTED, 0.0


# ---------------------------------------------------------------------------
# USDA FoodData Central
# ---------------------------------------------------------------------------

ENERGY_KCAL_IDS = (1008, 2047, 2048)
PROTEIN_IDS = (1003,)
FAT_IDS = (1004, 1085)
CARB_IDS = (1005, 1050)


class USDAError(Exception):
    pass


def _nutrient(food: dict, ids: tuple[int, ...], unit: str | None = None) -> float | None:
    values: dict[int, float] = {}
    for n in food.get("foodNutrients", []):
        nid = n.get("nutrientId")
        if nid in ids and n.get("value") is not None:
            if unit and str(n.get("unitName", "")).upper() != unit:
                continue
            values[nid] = float(n["value"])
    for nid in ids:
        if nid in values:
            return values[nid]
    return None


def extract_macros(food: dict) -> dict[str, float | None]:
    return {
        "calories_per_100g": _nutrient(food, ENERGY_KCAL_IDS, unit="KCAL"),
        "protein_per_100g": _nutrient(food, PROTEIN_IDS),
        "fat_per_100g": _nutrient(food, FAT_IDS),
        "carbs_per_100g": _nutrient(food, CARB_IDS),
    }


class USDAProvider:
    """USDA FoodData Central search (values are per 100 g for Foundation, SR Legacy and FNDDS)."""

    name = "usda_fdc"
    SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

    def __init__(self, settings: Settings, session=None, sleep=time.sleep) -> None:
        import requests

        self.settings = settings
        self.session = session or requests.Session()
        self.sleep = sleep
        self.min_interval = 3600.0 / max(1, settings.usda_requests_per_hour)
        self._last_request = 0.0
        self.data_types = list(settings.usda_data_types)
        if settings.using_usda_demo_key:
            logger.warning(
                "USDA_API_KEY not set: using DEMO_KEY (heavily rate limited). "
                "Get a free key at https://fdc.nal.usda.gov/api-key-signup"
            )

    def _throttle(self) -> None:
        wait = self._last_request + self.min_interval - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last_request = time.monotonic()

    def _fetch(self, query: str) -> list[dict]:
        import requests

        def call() -> list[dict]:
            self._throttle()
            resp = self.session.post(
                self.SEARCH_URL,
                params={"api_key": self.settings.usda_api_key},
                json={"query": query, "dataType": self.data_types, "pageSize": self.settings.usda_page_size},
                timeout=self.settings.usda_timeout_s,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise USDAError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            remaining = resp.headers.get("X-Ratelimit-Remaining")
            if remaining is not None and remaining.isdigit() and int(remaining) < 5:
                logger.warning("USDA rate limit nearly exhausted (%s left); pausing 10 minutes", remaining)
                self.sleep(600)
            return resp.json().get("foods", [])

        return with_retries(
            call,
            retry_on=(USDAError, requests.ConnectionError, requests.Timeout),
            max_attempts=self.settings.max_retries,
            base_s=max(self.settings.backoff_base_s, 5.0),
            max_s=max(self.settings.backoff_max_s, 300.0),
            description=f"USDA search {query!r}",
            sleep=self.sleep,
        )

    def rank_candidates(self, query: str, foods: list[dict]) -> list[tuple[MatchScore, dict, dict]]:
        ranked = []
        for food in foods:
            macros = extract_macros(food)
            if any(v is None for v in macros.values()):
                continue
            ms = match_score(query, food.get("description", ""))
            ranked.append((ms, food, macros))
        ranked.sort(
            key=lambda r: (
                -r[0].score,
                r[0].num_tokens,
                DATA_TYPE_PRIORITY.get(r[1].get("dataType", ""), 9),
            )
        )
        return ranked

    def search(self, food_name: str) -> NutritionResult | None:
        foods = self._fetch(food_name)
        ranked = self.rank_candidates(food_name, foods)
        if not ranked:
            return None
        ms, food, macros = ranked[0]
        source, confidence = score_to_source_confidence(ms.score, self.settings)
        flags = []
        # Same words rarely mean the same recipe for mixed dishes ("pork with bean" vs US "Pork and beans").
        if source == SOURCE_DB_DIRECT and is_mixed_dish(food_name):
            source, confidence = SOURCE_DB_APPROX, min(confidence, 0.89)
            flags.append("mixed_dish_database_match")
        return NutritionResult(
            normalized_name=food_name,
            source=source,
            confidence=confidence,
            matched_food_name=food.get("description"),
            fdc_id=food.get("fdcId"),
            match_score=round(ms.score, 3),
            query=food_name,
            notes=f"{self.name}:{food.get('dataType')}",
            flags=flags,
            created_at=utc_now_iso(),
            **macros,
        )


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------

OVERRIDE_COLUMNS = (
    "normalized_food_name", "calories_per_100g", "carbs_per_100g", "protein_per_100g", "fat_per_100g",
)


class ManualOverrideProvider:
    """Hand-curated per-100 g values. Read-only: the pipeline never writes this file."""

    name = "manual_override"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, NutritionResult] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        df = pd.read_csv(self.path, dtype={"normalized_food_name": str})
        missing = [c for c in OVERRIDE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{self.path} is missing columns: {missing}")
        for row in df.itertuples(index=False):
            name = normalize_food_name(str(row.normalized_food_name))
            values = {c: pd.to_numeric(getattr(row, c), errors="coerce") for c in OVERRIDE_COLUMNS[1:]}
            if not name or any(pd.isna(v) for v in values.values()):
                logger.warning("Skipping incomplete manual override row: %s", row)
                continue
            notes = getattr(row, "notes", None)
            self.entries[name] = NutritionResult(
                normalized_name=name,
                source=SOURCE_MANUAL,
                confidence=1.0,
                notes=None if pd.isna(notes) else str(notes),
                **{k: float(v) for k, v in values.items()},
            )
        logger.info("Loaded %d manual nutrition overrides from %s", len(self.entries), self.path)

    def search(self, food_name: str) -> NutritionResult | None:
        return self.entries.get(food_name)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@dataclass
class ResolveStats:
    unique_foods: int = 0
    from_override: int = 0
    from_cache: int = 0
    database_queries: int = 0
    database_failures: int = 0
    llm_requested: int = 0
    llm_failed: int = 0


def unresolved(name: str, reason: str, base: NutritionResult | None = None) -> NutritionResult:
    data = base.model_dump() if base is not None else {}
    data.update(
        normalized_name=name,
        calories_per_100g=None,
        carbs_per_100g=None,
        protein_per_100g=None,
        fat_per_100g=None,
        source=SOURCE_UNRESOLVED,
        confidence=0.0,
        flags=sorted(set(data.get("flags", []) + [reason])),
    )
    return NutritionResult(**data)


class NutritionResolver:
    def __init__(
        self,
        cache: JsonCache,
        settings: Settings,
        providers: list[NutritionProvider],
        overrides: ManualOverrideProvider | None = None,
        estimator: NutritionEstimator | None = None,
        refresh: bool = False,
        journal: ProgressJournal = NULL_JOURNAL,
    ) -> None:
        self.journal = journal
        self.cache = cache
        self.settings = settings
        self.providers = providers
        self.overrides = overrides
        self.estimator = estimator
        self.refresh = refresh
        self.stats = ResolveStats()

    def _from_cache(self, name: str) -> NutritionResult | None:
        entry = self.cache.get(name)
        if entry is None or self.refresh:
            return None
        return NutritionResult(**entry)

    def _query_database(self, name: str) -> NutritionResult | None:
        """Return an accepted database result, or a db_miss marker; None if every provider errored."""
        best_rejected: NutritionResult | None = None
        errored = 0
        for provider in self.providers:
            self.stats.database_queries += 1
            try:
                result = provider.search(name)
            except Exception as exc:
                errored += 1
                self.stats.database_failures += 1
                logger.error("Nutrition lookup failed for %r via %s: %s", name, getattr(provider, "name", provider), exc)
                continue
            if result is None:
                continue
            if result.source in DATABASE_SOURCES and result.has_macros:
                return result
            if best_rejected is None or (result.match_score or 0) > (best_rejected.match_score or 0):
                best_rejected = result
        if self.providers and errored == len(self.providers):
            return None
        miss = NutritionResult(
            normalized_name=name,
            calories_per_100g=None,
            carbs_per_100g=None,
            protein_per_100g=None,
            fat_per_100g=None,
            source=SOURCE_DB_MISS,
            confidence=0.0,
            query=name,
            created_at=utc_now_iso(),
        )
        if best_rejected is not None:
            miss.db_candidate_name = best_rejected.matched_food_name
            miss.db_candidate_fdc_id = best_rejected.fdc_id
            miss.db_candidate_score = best_rejected.match_score
        return miss

    def resolve(self, names: Iterable[str], use_llm_fallback: bool = True) -> dict[str, NutritionResult]:
        unique = sorted({n for n in names if n})
        self.stats.unique_foods = len(unique)
        results: dict[str, NutritionResult] = {}
        misses: dict[str, NutritionResult] = {}

        to_query: list[str] = []
        for name in unique:
            override = self.overrides.search(name) if self.overrides else None
            if override is not None:
                results[name] = override
                self.stats.from_override += 1
                continue
            cached = self._from_cache(name)
            if cached is not None and cached.source in FINAL_SOURCES:
                results[name] = cached
                self.stats.from_cache += 1
            elif cached is not None and cached.source == SOURCE_DB_MISS:
                misses[name] = cached
            else:
                to_query.append(name)

        logger.info(
            "Nutrition: %d unique foods (%d manual overrides, %d cached, %d known database misses, %d to look up)",
            len(unique), self.stats.from_override, self.stats.from_cache, len(misses), len(to_query),
        )

        self.journal.log("nutrition_start", unique_foods=len(unique), cached=self.stats.from_cache,
                         known_misses=len(misses), to_look_up=len(to_query))
        try:
            for i, name in enumerate(to_query, start=1):
                result = self._query_database(name)
                if result is None:
                    results[name] = unresolved(name, "database_lookup_error")
                    continue
                self.cache.set(name, result.model_dump())
                if result.source == SOURCE_DB_MISS:
                    misses[name] = result
                else:
                    results[name] = result
                if i % 25 == 0 or i == len(to_query):
                    logger.info("Database lookups: %d/%d", i, len(to_query))
                    self.journal.log("database_progress", done=i, of=len(to_query), misses=len(misses))
        finally:
            self.cache.flush()
            self.cache.snapshot("database")

        if misses:
            if use_llm_fallback and self.estimator is not None:
                self._run_llm_fallback(misses, results)
            else:
                for name, miss in misses.items():
                    results[name] = unresolved(name, "llm_fallback_skipped", miss)
        return results

    def _run_llm_fallback(self, misses: dict[str, NutritionResult], results: dict[str, NutritionResult]) -> None:
        logger.info("LLM fallback: estimating %d foods not confidently matched in the database", len(misses))
        self.stats.llm_requested = len(misses)
        self.journal.log("llm_fallback_start", foods=len(misses))
        done = 0

        def save_batch(batch: dict[str, NutritionResult]) -> None:
            nonlocal done
            for name, est in batch.items():
                miss = misses.get(name)
                if miss is None:
                    continue
                est = est.model_copy(
                    update={
                        "db_candidate_name": miss.db_candidate_name,
                        "db_candidate_fdc_id": miss.db_candidate_fdc_id,
                        "db_candidate_score": miss.db_candidate_score,
                    }
                )
                self.cache.set(name, est.model_dump())
                results[name] = est
            self.cache.checkpoint()
            done += len(batch)
            self.journal.log("llm_fallback_batch_done", estimated=done, of=len(misses))

        try:
            self.estimator.estimate(sorted(misses), on_batch=save_batch)
        finally:
            self.cache.flush()
            self.cache.snapshot("llm_fallback")
        for name, miss in misses.items():
            if name not in results:
                self.stats.llm_failed += 1
                results[name] = unresolved(name, "llm_fallback_failed", miss)
        self.journal.log("llm_fallback_done", estimated=done, failed=self.stats.llm_failed)


def count_sources(results: dict[str, NutritionResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.source] = counts.get(r.source, 0) + 1
    return counts
