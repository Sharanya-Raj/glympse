"""End-to-end orchestration: load -> parse -> normalize/dedupe -> resolve nutrition -> scale -> write."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .cache import JsonCache, ProgressJournal, RunLock, atomic_write_json, utc_now_iso
from .config import PROJECT_ROOT, Settings
from .llm_client import LLMClient, UsageTracker
from .llm_estimator import LLMNutritionEstimator
from .macro_calculator import CONFIDENCE_STRATEGIES, aggregate_meal, scale_macros
from .models import (
    SOURCE_UNRESOLVED,
    SUCCESS_STATUSES,
    MEAL_STATUS_PARTIAL,
    FoodResult,
    MealResult,
    NutritionResult,
    ParsedMeal,
)
from .normalize_food import normalize_food_name
from .nutrition_lookup import (
    ManualOverrideProvider,
    NutritionProvider,
    NutritionResolver,
    USDAProvider,
    count_sources,
    unresolved,
)
from .parse_meals import PARSER_LLM, MealParser
from .units import convert_to_grams, normalize_unit

logger = logging.getLogger(__name__)

MEAL_COLUMNS = [
    "meal_id", "meal_text", "calories", "carbs_g", "protein_g", "fat_g", "confidence",
    "num_food_items", "num_database_matches", "num_llm_estimates",
    "num_db_direct", "num_db_approximate", "num_manual_overrides", "num_unresolved_items",
    "total_grams", "status", "parse_warnings",
]
FOOD_COLUMNS = [
    "meal_id", "original_food_text", "normalized_food_name", "amount", "unit", "estimated_grams",
    "calories", "carbs_g", "protein_g", "fat_g", "nutrition_source", "nutrition_confidence",
    "unit_conversion_confidence",
    "original_food_name", "preparation", "matched_food_name", "fdc_id", "match_score",
    "reasoning_summary", "nutrition_flags", "parser", "parser_model", "estimator_model",
]


@dataclass
class PipelineOptions:
    input_path: Path
    output_path: Path
    breakdown_path: Path | None = None
    metadata_path: Path | None = None
    batch_size: int | None = None
    limit: int | None = None
    resume: bool = True
    force_reparse: bool = False
    refresh_nutrition: bool = False
    skip_llm_fallback: bool = False
    id_column: str | None = None
    text_column: str | None = None
    confidence_strategy: str | None = None
    parser: str = PARSER_LLM


@dataclass
class RunSummary:
    processed_meals: int = 0
    meals_from_cache: int = 0
    successful: int = 0
    partial: int = 0
    failed: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    unique_foods: int = 0
    nutrition_sources: dict[str, int] = field(default_factory=dict)
    food_item_sources: dict[str, int] = field(default_factory=dict)
    mean_meal_confidence: float | None = None
    llm_usage: dict = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def load_meals(path: Path, id_column: str, text_column: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={id_column: str}, keep_default_na=False)
    missing = [c for c in (id_column, text_column) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; found {list(df.columns)}. "
                         "Use --id-column/--text-column to map your columns.")
    df[id_column] = df[id_column].astype(str).str.strip()
    df[text_column] = df[text_column].astype(str)
    dupes = df[id_column][df[id_column].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"Duplicate meal ids in {path}: {list(dupes[:10])}")
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def build_food_results(
    meal_id: str,
    parsed: ParsedMeal,
    nutrition: dict[str, NutritionResult],
    estimator_model: str | None,
) -> list[FoodResult]:
    results = []
    for food in parsed.foods:
        normalized = normalize_food_name(food.name) or normalize_food_name(food.original_text)
        nut = nutrition.get(normalized) or unresolved(normalized, "not_resolved")
        grams, conversion_conf = convert_to_grams(normalized, food.amount, food.unit)
        macros = scale_macros(nut if nut.has_macros else None, grams)
        flags = list(nut.flags)
        if grams is None:
            flags.append("unit_conversion_failed")
        results.append(
            FoodResult(
                meal_id=meal_id,
                original_food_text=food.original_text,
                original_food_name=food.name,
                normalized_food_name=normalized,
                preparation=food.preparation,
                amount=food.amount,
                unit=normalize_unit(food.unit) or food.unit,
                estimated_grams=None if grams is None else round(grams, 2),
                calories=None if macros["calories"] is None else round(macros["calories"], 2),
                carbs_g=None if macros["carbs_g"] is None else round(macros["carbs_g"], 3),
                protein_g=None if macros["protein_g"] is None else round(macros["protein_g"], 3),
                fat_g=None if macros["fat_g"] is None else round(macros["fat_g"], 3),
                nutrition_source=nut.source,
                nutrition_confidence=nut.confidence,
                unit_conversion_confidence=conversion_conf,
                matched_food_name=nut.matched_food_name or nut.db_candidate_name,
                fdc_id=nut.fdc_id,
                match_score=nut.match_score if nut.match_score is not None else nut.db_candidate_score,
                reasoning_summary=nut.reasoning_summary,
                nutrition_flags="; ".join(sorted(set(flags))) or None,
                parser=parsed.parser,
                parser_model=parsed.model,
                estimator_model=nut.model or (estimator_model if nut.source == "llm_estimate" else None),
            )
        )
    return results


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_pipeline(
    options: PipelineOptions,
    settings: Settings,
    *,
    llm_client=None,
    providers: list[NutritionProvider] | None = None,
) -> RunSummary:
    """`llm_client` / `providers` override the real OpenAI-compatible client and USDA (used by tests)."""
    with RunLock(settings.cache_dir / ".run.lock"):
        return _run_pipeline(options, settings, llm_client=llm_client, providers=providers)


def _run_pipeline(options, settings, *, llm_client, providers) -> RunSummary:
    id_col = options.id_column or settings.id_column
    text_col = options.text_column or settings.text_column
    strategy = options.confidence_strategy or settings.confidence_strategy
    if strategy not in CONFIDENCE_STRATEGIES:
        raise ValueError(f"Unknown confidence strategy {strategy!r}; choose from {sorted(CONFIDENCE_STRATEGIES)}")
    force_reparse = options.force_reparse or not options.resume
    refresh_nutrition = options.refresh_nutrition or not options.resume

    df = load_meals(options.input_path, id_col, text_col, options.limit)
    logger.info("Loaded %d meals from %s", len(df), options.input_path)
    journal = ProgressJournal(settings.journal_path)
    journal.log("run_start", input=str(options.input_path), meals=len(df), provider=settings.llm_provider,
                model=settings.llm_model, parser=options.parser, limit=options.limit)
    logger.info("Run id %s (progress journal: %s)", journal.run_id, settings.journal_path)

    usage = UsageTracker(settings.model_input_cost_per_million, settings.model_output_cost_per_million)
    llm: LLMClient | None = None
    if llm_client is not None or settings.llm_api_key:
        llm = LLMClient(settings, usage, client=llm_client)
        logger.info("LLM: %s model %s", settings.llm_provider, settings.llm_model)
    else:
        logger.warning("No LLM API key set: LLM parsing and LLM fallback are unavailable")

    snapshot_kwargs = {
        "snapshot_dir": settings.snapshot_dir,
        "snapshot_every": settings.snapshot_every,
        "snapshot_keep": settings.snapshot_keep,
        "snapshot_min_interval_s": settings.snapshot_min_interval_s,
    }

    # 1. Parse (the parser flushes explicitly after every LLM batch)
    parse_cache = JsonCache(settings.parsed_cache_path, autosave_every=settings.parse_autosave_every,
                            **snapshot_kwargs)
    parser = MealParser(parse_cache, settings, llm=llm, mode=options.parser, batch_size=options.batch_size,
                        journal=journal)
    meals = list(zip(df[id_col], df[text_col]))
    outcome = parser.parse_meals(meals, force=force_reparse)

    # 2. Normalize + dedupe
    unique_names = {
        normalize_food_name(f.name) or normalize_food_name(f.original_text)
        for pm in outcome.parsed.values()
        for f in pm.foods
    }
    unique_names.discard("")

    # 3. Resolve nutrition
    nutrition_cache = JsonCache(
        settings.nutrition_cache_path, autosave_every=settings.nutrition_checkpoint_every, **snapshot_kwargs
    )
    overrides = ManualOverrideProvider(settings.overrides_path)
    estimator = LLMNutritionEstimator(llm, settings) if llm is not None else None
    if providers is None:
        providers = [USDAProvider(settings)]
    resolver = NutritionResolver(
        nutrition_cache, settings, providers, overrides=overrides, estimator=estimator,
        refresh=refresh_nutrition, journal=journal,
    )
    nutrition = resolver.resolve(unique_names, use_llm_fallback=not options.skip_llm_fallback)

    # 4. Scale + aggregate
    meal_rows: list[MealResult] = []
    food_rows: list[FoodResult] = []
    estimator_model = estimator.model if estimator else None
    for meal_id, text in meals:
        parsed = outcome.parsed.get(meal_id)
        if parsed is None:
            meal_rows.append(aggregate_meal(meal_id, text, [], strategy, parse_failed=True,
                                            parse_warnings=[outcome.failures.get(meal_id, "not parsed")]))
            continue
        foods = build_food_results(meal_id, parsed, nutrition, estimator_model)
        food_rows.extend(foods)
        meal_rows.append(aggregate_meal(meal_id, text, foods, strategy, parse_warnings=parsed.warnings))

    # 5. Write outputs
    out_path = options.output_path
    breakdown_path = options.breakdown_path or out_path.with_name("food_breakdown.csv")
    metadata_path = options.metadata_path or out_path.with_name("run_metadata.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_path.parent.mkdir(parents=True, exist_ok=True)

    meals_df = pd.DataFrame([m.model_dump() for m in meal_rows], columns=MEAL_COLUMNS)
    passthrough = [c for c in df.columns if c not in (id_col, text_col)]
    if passthrough:
        meals_df = pd.concat([meals_df, df[passthrough].reset_index(drop=True)], axis=1)
    meals_df.to_csv(out_path, index=False, encoding="utf-8")
    pd.DataFrame([f.model_dump() for f in food_rows], columns=FOOD_COLUMNS).to_csv(
        breakdown_path, index=False, encoding="utf-8"
    )

    # 6. Summary + metadata
    statuses = [m.status for m in meal_rows]
    successful = [m for m in meal_rows if m.status in SUCCESS_STATUSES]
    confidences = [m.confidence for m in successful if m.confidence is not None]
    food_sources: dict[str, int] = {}
    for f in food_rows:
        food_sources[f.nutrition_source] = food_sources.get(f.nutrition_source, 0) + 1
    summary = RunSummary(
        processed_meals=len(meal_rows),
        meals_from_cache=outcome.cached_meals,
        successful=len(successful),
        partial=statuses.count(MEAL_STATUS_PARTIAL),
        failed=len(meal_rows) - len(successful),
        status_counts={s: statuses.count(s) for s in sorted(set(statuses))},
        unique_foods=len(nutrition),
        nutrition_sources=count_sources(nutrition),
        food_item_sources=food_sources,
        mean_meal_confidence=round(sum(confidences) / len(confidences), 3) if confidences else None,
        llm_usage=usage.snapshot().model_dump(),
        outputs={"meals": str(out_path), "food_breakdown": str(breakdown_path), "metadata": str(metadata_path)},
    )
    atomic_write_json(
        metadata_path,
        {
            "run_at": utc_now_iso(),
            "git_commit": _git_commit(),
            "input": str(options.input_path),
            "options": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(options).items()},
            "run_id": journal.run_id,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "parser_prompt_version": settings.parser_prompt_version,
            "estimator_prompt_version": settings.estimator_prompt_version,
            "usda_data_types": list(settings.usda_data_types),
            "match_thresholds": {
                "direct": settings.match_direct_threshold,
                "approximate": settings.match_approximate_threshold,
            },
            "confidence_strategy": strategy,
            "prices_per_million": {
                "input": settings.model_input_cost_per_million,
                "output": settings.model_output_cost_per_million,
            },
            "nutrition_resolution": asdict(resolver.stats),
            "summary": asdict(summary),
        },
    )
    archive = out_path.parent / "history" / journal.run_id
    archive.mkdir(parents=True, exist_ok=True)
    for path in (out_path, breakdown_path, metadata_path):
        shutil.copy2(path, archive / path.name)
    summary.outputs["archive"] = str(archive)
    journal.log("run_done", successful=summary.successful, failed=summary.failed, unique_foods=summary.unique_foods,
                nutrition_sources=summary.nutrition_sources, llm_usage=summary.llm_usage, archive=str(archive))
    if SOURCE_UNRESOLVED in summary.nutrition_sources:
        logger.warning("%d unique foods remain unresolved (see nutrition_flags in the breakdown CSV)",
                       summary.nutrition_sources[SOURCE_UNRESOLVED])
    return summary


SOURCE_LABELS = [
    ("manual_override", "Manual override"),
    ("database_direct", "Database (direct)"),
    ("database_approximate", "Database (approximate)"),
    ("llm_estimate", "LLM fallback"),
    ("unresolved", "Unresolved"),
]


def format_summary(summary: RunSummary) -> str:
    db_total = summary.nutrition_sources.get("database_direct", 0) + summary.nutrition_sources.get(
        "database_approximate", 0
    )
    lines = [
        f"Processed meals: {summary.processed_meals}",
        f"Successful: {summary.successful} (of which partial: {summary.partial})",
        f"Failed: {summary.failed}",
        f"Meal statuses: {summary.status_counts}",
        "",
        f"Unique foods: {summary.unique_foods}",
        "",
        "Nutrition sources (unique foods):",
        f"  Database: {db_total}",
    ]
    for key, label in SOURCE_LABELS:
        lines.append(f"  {label}: {summary.nutrition_sources.get(key, 0)}")
    conf = summary.mean_meal_confidence
    lines += ["", f"Mean meal confidence: {conf:.2f}" if conf is not None else "Mean meal confidence: n/a"]
    usage = summary.llm_usage
    cost = usage.get("estimated_cost_usd")
    lines += [
        "",
        f"LLM requests: {usage.get('requests', 0)} (failed attempts: {usage.get('failed_requests', 0)})",
        f"LLM tokens: {usage.get('input_tokens', 0)} input, {usage.get('output_tokens', 0)} output",
        f"Estimated API cost: ${cost:.4f}" if cost is not None else "Estimated API cost: unknown (no price configured)",
        "",
        f"Outputs: {summary.outputs.get('meals')}",
        f"         {summary.outputs.get('food_breakdown')}",
        f"         {summary.outputs.get('metadata')}",
        f"Run archive: {summary.outputs.get('archive')}",
    ]
    return "\n".join(lines)
