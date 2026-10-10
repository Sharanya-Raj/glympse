"""Convert free-text meal descriptions into structured food items.

Two parsers share the same cache/validation path:
- "llm": batched OpenAI structured-output parsing (default)
- "rule": deterministic regex parser for "<food> <amount> <unit>" records (offline testing / comparison)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from .cache import NULL_JOURNAL, JsonCache, ProgressJournal, utc_now_iso
from .config import Settings
from .llm_client import FailureBreaker, LLMClient, LLMUnavailableError
from .models import FoodItem, ParsedMeal

logger = logging.getLogger(__name__)

PARSER_LLM = "llm"
PARSER_RULE = "rule"

SYSTEM_PROMPT = """You convert free-text meal records from a clinical CGM study in Shanghai \
(Chinese hospital and home meals, translated to English) into structured food items.

Rules:
1. Return exactly one entry per input meal, using the exact meal_id given.
2. Split the meal into the food items it lists. Items are separated by semicolons, line breaks, \
or simply follow each other as "<food> <amount> <unit>".
3. original_text: copy the exact substring of the meal text for that item, including its amount \
and unit. Do not fix spelling or spacing in original_text.
4. name: the food name in lowercase English without the amount, keeping preparation words \
(e.g. "dry-fried pomfret", "braised duck wing in brown sauce"). You may fix obvious typos in name only.
5. amount: the number exactly as written (e.g. 96.2), or null if no amount is given. \
unit: the unit as written (g, ml, kg, ...), or null if none.
6. preparation: the cooking/preparation method if stated (e.g. "steamed", "dry-fried", \
"braised in brown sauce", "sliced"), otherwise null.
7. Never invent foods, amounts, or units that are not in the text. Do not add oil, sauces or sides.
8. Keep mixed dishes and lists that share one amount as a single item: \
"Lotus nut, red bean, poria coco and orange peel 30 g" is ONE item with amount 30. \
Never split a shared amount among ingredients.
9. If the text is not a food record (e.g. "data not available"), return an empty foods list."""


class _FoodOut(BaseModel):
    original_text: str
    name: str
    amount: float | None
    unit: str | None
    preparation: str | None


class _MealOut(BaseModel):
    meal_id: str
    foods: list[_FoodOut]


class ParsedMealBatch(BaseModel):
    meals: list[_MealOut]


def _collapse(text: str) -> str:
    return " ".join(text.split())


def meal_text_key(text: str) -> str:
    return hashlib.sha256(_collapse(text).encode("utf-8")).hexdigest()


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def validate_foods(meal_text: str, foods: list[FoodItem]) -> tuple[list[str], list[str]]:
    """Return (hard_errors, soft_warnings) for a parsed meal."""
    hard: list[str] = []
    soft: list[str] = []
    if _collapse(meal_text) and not foods:
        hard.append("no foods parsed from non-empty text")
    haystack = _collapse(meal_text).casefold()
    for food in foods:
        if not food.name.strip():
            hard.append(f"empty name for {food.original_text!r}")
        if _collapse(food.original_text).casefold() not in haystack:
            soft.append(f"original_text not found verbatim: {food.original_text!r}")
        numbers = [float(n) for n in _NUMBER_RE.findall(food.original_text)]
        if food.amount is not None and not any(abs(n - food.amount) < 1e-6 for n in numbers):
            soft.append(f"amount {food.amount} not present in {food.original_text!r}")
        if food.amount is not None and food.amount < 0:
            hard.append(f"negative amount in {food.original_text!r}")
    return hard, soft


# ---------------------------------------------------------------------------
# Rule-based parser
# ---------------------------------------------------------------------------

_UNIT_PATTERN = r"(?:kg|g|grams?|mg|ml|mL|millilit(?:er|re)s?|l|L|lit(?:er|re)s?|oz|ounces?)"
_ITEM_RE = re.compile(
    rf"(?P<name>.*?)\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>{_UNIT_PATTERN})\b\.?",
    re.IGNORECASE,
)
_PREP_WORDS = (
    "stir-fried", "stir fried", "dry-fried", "dry fried", "deep-fried", "deep fried", "pan-fried",
    "fried", "steamed", "braised", "boiled", "stewed", "roasted", "grilled", "baked", "scrambled",
    "poached", "smoked", "pickled", "marinated", "sliced", "shredded", "minced", "diced", "mashed",
)
_PREP_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _PREP_WORDS) + r")\b", re.IGNORECASE)


def rule_parse_meal(meal_text: str) -> list[FoodItem]:
    foods: list[FoodItem] = []
    for segment in re.split(r"[;\n]+", meal_text):
        segment = segment.strip(" ,")
        if not segment:
            continue
        end = 0
        for match in _ITEM_RE.finditer(segment):
            name = match.group("name").strip(" ,")
            if not name:
                continue
            preps = _PREP_RE.findall(name)
            foods.append(
                FoodItem(
                    original_text=match.group(0).strip(" ,"),
                    name=name.lower(),
                    amount=float(match.group("amount")),
                    unit=match.group("unit"),
                    preparation=", ".join(p.lower() for p in preps) or None,
                )
            )
            end = match.end()
        leftover = segment[end:].strip(" ,")
        if leftover:
            foods.append(FoodItem(original_text=leftover, name=leftover.lower(), amount=None, unit=None))
    return foods


# ---------------------------------------------------------------------------
# Parser with caching
# ---------------------------------------------------------------------------


@dataclass
class ParseOutcome:
    parsed: dict[str, ParsedMeal] = field(default_factory=dict)  # meal_id -> ParsedMeal
    failures: dict[str, str] = field(default_factory=dict)  # meal_id -> reason
    cached_meals: int = 0
    newly_parsed_texts: int = 0


class MealParser:
    def __init__(
        self,
        cache: JsonCache,
        settings: Settings,
        llm: LLMClient | None = None,
        mode: str = PARSER_LLM,
        batch_size: int | None = None,
        journal: ProgressJournal = NULL_JOURNAL,
    ) -> None:
        if mode not in (PARSER_LLM, PARSER_RULE):
            raise ValueError(f"Unknown parser {mode!r}")
        self.journal = journal
        self.cache = cache
        self.settings = settings
        self.llm = llm
        self.mode = mode
        self.batch_size = max(1, batch_size or settings.batch_size)

    def _cache_key(self, text: str) -> str:
        return f"{self.mode}:{meal_text_key(text)}"

    def _store(self, key: str, text: str, foods: list[FoodItem], warnings: list[str], meal_ids: list[str]) -> None:
        self.cache.set(
            key,
            {
                "meal_text": text,
                "foods": [f.model_dump() for f in foods],
                "parser": self.mode,
                "model": self.llm.model if self.mode == PARSER_LLM and self.llm else None,
                "prompt_version": self.settings.parser_prompt_version if self.mode == PARSER_LLM else None,
                "parsed_at": utc_now_iso(),
                "warnings": warnings,
                "meal_ids": sorted(set(meal_ids)),
            },
        )

    def _to_parsed(self, meal_id: str, entry: dict) -> ParsedMeal:
        return ParsedMeal(
            meal_id=meal_id,
            foods=[FoodItem(**f) for f in entry["foods"]],
            parser=entry.get("parser"),
            model=entry.get("model"),
            prompt_version=entry.get("prompt_version"),
            warnings=entry.get("warnings", []),
        )

    def parse_meals(self, meals: list[tuple[str, str]], force: bool = False) -> ParseOutcome:
        outcome = ParseOutcome()
        ids_by_key: dict[str, list[str]] = {}
        text_by_key: dict[str, str] = {}
        for meal_id, text in meals:
            key = self._cache_key(text)
            ids_by_key.setdefault(key, []).append(meal_id)
            text_by_key[key] = text

        pending: list[str] = []
        for key, meal_ids in ids_by_key.items():
            entry = self.cache.get(key)
            if entry is not None and not force:
                outcome.cached_meals += len(meal_ids)
                for mid in meal_ids:
                    outcome.parsed[mid] = self._to_parsed(mid, entry)
            elif not _collapse(text_by_key[key]):
                for mid in meal_ids:
                    outcome.parsed[mid] = ParsedMeal(meal_id=mid, foods=[], parser=self.mode)
            else:
                pending.append(key)

        logger.info(
            "Parsing: %d meals total, %d already cached, %d unique texts to parse with %s parser",
            len(meals), outcome.cached_meals, len(pending), self.mode,
        )

        if self.mode == PARSER_RULE:
            self._parse_rule(pending, text_by_key, ids_by_key, outcome)
        else:
            self._parse_llm(pending, text_by_key, ids_by_key, outcome)
        self.cache.flush()
        return outcome

    def _accept(self, key, text, foods, warnings, ids_by_key, outcome) -> None:
        self._store(key, text, foods, warnings, ids_by_key[key])
        outcome.newly_parsed_texts += 1
        for mid in ids_by_key[key]:
            outcome.parsed[mid] = self._to_parsed(mid, self.cache.get(key))

    def _fail(self, key, reason, ids_by_key, outcome, quiet: bool = False) -> None:
        for mid in ids_by_key[key]:
            outcome.failures[mid] = reason
        if not quiet:
            logger.error("Parsing failed for meal(s) %s: %s", ", ".join(ids_by_key[key]), reason)

    def _parse_rule(self, pending, text_by_key, ids_by_key, outcome) -> None:
        for key in pending:
            text = text_by_key[key]
            foods = rule_parse_meal(text)
            hard, soft = validate_foods(text, foods)
            if hard:
                self._fail(key, "; ".join(hard), ids_by_key, outcome)
            else:
                self._accept(key, text, foods, soft, ids_by_key, outcome)
        if pending:
            self.cache.flush()
            self.cache.snapshot("parse")

    def _request(self, items: list[tuple[str, str]]) -> dict[str, list[FoodItem]]:
        """items: (local_id, meal_text). Returns local_id -> foods for the meals the model returned."""
        payload = [{"meal_id": local_id, "meal_text": text} for local_id, text in items]
        result = self.llm.parse(
            system=SYSTEM_PROMPT,
            user="Parse these meals:\n" + json.dumps(payload, ensure_ascii=False),
            response_format=ParsedMealBatch,
            purpose="parse_meals",
        )
        out: dict[str, list[FoodItem]] = {}
        for meal in result.meals:
            out[meal.meal_id] = [FoodItem(**f.model_dump()) for f in meal.foods]
        return out

    def _parse_llm(self, pending, text_by_key, ids_by_key, outcome) -> None:
        if pending and self.llm is None:
            raise RuntimeError(
                f"{len(pending)} meals need LLM parsing but no LLM API key is set "
                "(set OPENAI_API_KEY or GOOGLE_API_KEY in .env, or use --parser rule for offline testing)"
            )
        batches = [pending[i : i + self.batch_size] for i in range(0, len(pending), self.batch_size)]
        breaker = FailureBreaker(self.settings.llm_max_consecutive_failures)
        self.journal.log("parse_start", pending_texts=len(pending), batches=len(batches), cached_meals=outcome.cached_meals)
        try:
            for b_idx, batch in enumerate(batches, start=1):
                try:
                    self._parse_batch(b_idx, len(batches), batch, breaker, text_by_key, ids_by_key, outcome)
                except LLMUnavailableError as exc:
                    logger.error("%s", exc)
                    for later in batches[b_idx:]:
                        for key in later:
                            self._fail(key, "skipped: LLM unavailable", ids_by_key, outcome, quiet=True)
                    self.journal.log("parse_aborted", batch=b_idx, reason=str(exc))
                    break
        finally:
            self.cache.flush()
            self.cache.snapshot("parse")
        self.journal.log("parse_done", parsed_texts=outcome.newly_parsed_texts, failed_meals=len(outcome.failures))

    def _parse_batch(self, b_idx, n_batches, batch, breaker, text_by_key, ids_by_key, outcome) -> None:
        local = {f"m{i + 1}": key for i, key in enumerate(batch)}
        try:
            returned = self._request([(lid, text_by_key[key]) for lid, key in local.items()])
        except Exception as exc:  # batch failed after retries: log and move on
            logger.error("Parse batch %d/%d failed after retries: %s", b_idx, n_batches, exc)
            for key in batch:
                self._fail(key, f"batch request failed: {exc}", ids_by_key, outcome)
            self.journal.log("parse_batch_failed", batch=b_idx, of=n_batches, error=str(exc)[:500])
            breaker.failure("parse")
            return
        breaker.success()
        self._handle_batch_response(b_idx, n_batches, local, returned, text_by_key, ids_by_key, outcome)

    def _handle_batch_response(self, b_idx, n_batches, local, returned, text_by_key, ids_by_key, outcome) -> None:
        retry_individually: list[str] = []
        for lid, key in local.items():
            foods = returned.get(lid)
            if foods is None:
                retry_individually.append(key)
                continue
            hard, soft = validate_foods(text_by_key[key], foods)
            if hard or soft:
                logger.warning("Meal %s failed batch validation (%s); retrying alone", lid, "; ".join(hard + soft))
                retry_individually.append(key)
            else:
                self._accept(key, text_by_key[key], foods, [], ids_by_key, outcome)

        for key in retry_individually:
            self._parse_single(key, text_by_key, ids_by_key, outcome)

        self.cache.checkpoint()
        logger.info(
            "Parse batch %d/%d done (%d texts parsed so far, %d meal failures); checkpoint written",
            b_idx, n_batches, outcome.newly_parsed_texts, len(outcome.failures),
        )
        self.journal.log(
            "parse_batch_done", batch=b_idx, of=n_batches,
            parsed_texts=outcome.newly_parsed_texts, failed_meals=len(outcome.failures),
        )

    def _parse_single(self, key, text_by_key, ids_by_key, outcome) -> None:
        text = text_by_key[key]
        try:
            returned = self._request([("m1", text)])
        except Exception as exc:
            self._fail(key, f"single request failed: {exc}", ids_by_key, outcome)
            return
        foods = returned.get("m1")
        if foods is None and len(returned) == 1:
            foods = next(iter(returned.values()))
        if foods is None:
            self._fail(key, "model did not return the meal", ids_by_key, outcome)
            return
        hard, soft = validate_foods(text, foods)
        if hard:
            self._fail(key, "; ".join(hard), ids_by_key, outcome)
            return
        if soft:
            logger.warning("Accepting meal %s with parse warnings: %s", ids_by_key[key], "; ".join(soft))
        self._accept(key, text, foods, soft, ids_by_key, outcome)
