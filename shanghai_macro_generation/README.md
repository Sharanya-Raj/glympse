# Shanghai meal macro generation

Estimates calories, carbohydrates, protein and fat for the ~3,500 free-text meal records in the
Shanghai T2DM CGM dataset, so the meals can be used as inputs to a glucose prediction model.

Every estimate keeps its provenance: the original meal text, the original food phrase, the parsed
amount and unit, where each food's nutrition came from (manual override, USDA direct match,
approximate USDA match or LLM estimate), the confidence of each step, and the model and prompt
version that produced it.

> **LLM-derived macros are estimates, not nutritional ground truth.** See [Limitations](#limitations).

## Architecture

```text
raw meal description
  -> LLM parsing (batched, structured JSON, validated)        src/parse_meals.py
  -> structured food items                                    src/models.py
  -> food name normalization + deduplication                  src/normalize_food.py
  -> manual override -> cache -> USDA FoodData Central         src/nutrition_lookup.py
  -> LLM per-100 g estimate if the database match is weak     src/llm_estimator.py
  -> amount -> grams (mass units, densities for volumes)      src/units.py
  -> scale per-100 g values, sum per meal, meal confidence    src/macro_calculator.py
  -> CSV outputs + run metadata                               src/pipeline.py
```

Other modules: `src/config.py` (all settings, prices, thresholds), `src/llm_client.py` (OpenAI
structured outputs, retries, token/cost tracking), `src/cache.py` (atomic JSON checkpoints),
`src/retry.py` (exponential backoff).

```text
shanghai_macro_generation/
  src/            pipeline modules (above)
  scripts/        prepare_shanghai_input.py, process_meals.py, restore_snapshot.py
  tests/          unit tests (no network; OpenAI and USDA are mocked)
  data/raw/       pipeline input (shanghai_meals.csv, example_meals.csv)
  data/processed/ outputs
  data/manual_nutrition_overrides.csv
  cache/          parsed_meals.json, nutrition_cache.json, progress.jsonl, snapshots/ (created on first run)
  logs/           process_meals.log
```

## Installation

From the repository root (Python 3.10+):

```bash
uv pip install --python .venv/Scripts/python.exe -r shanghai_macro_generation/requirements.txt
# or, inside any virtualenv:
pip install -r shanghai_macro_generation/requirements.txt
```

All commands below are run from `shanghai_macro_generation/`.

## API keys

Copy `.env.example` to `.env` (in this folder; the repo-root `.env` is also read as a fallback).
Either Google Gemini or OpenAI can be the LLM:

```text
GOOGLE_API_KEY=...      # Gemini via Google's OpenAI-compatible endpoint (also read: GEMINI_API_KEY, GOOGLE_API_AI)
# OPENAI_API_KEY=sk-... # used instead if set (or force with LLM_PROVIDER=openai|gemini)
USDA_API_KEY=...        # free: https://fdc.nal.usda.gov/api-key-signup
```

With Gemini the default model is `gemini-2.5-flash` and requests are throttled to 8 per minute
(the free tier allows about 10). Raise `LLM_REQUESTS_PER_MINUTE` on a paid tier. Free-tier
Gemini requests may be used by Google to improve its products.

Without `USDA_API_KEY` the public `DEMO_KEY` is used, which allows only about 30 requests per hour.
Model prices live in `MODEL_PRICES` in `src/config.py` and can be overridden with
`MODEL_INPUT_COST_PER_MILLION` and `MODEL_OUTPUT_COST_PER_MILLION`. `.env` is git-ignored.

## Input format

A CSV with a meal id column and a meal text column (defaults `meal_id`, `meal_text`; change them with
`--id-column` / `--text-column` or in `src/config.py`). Any extra columns (patient id, timestamp, ...)
are carried through to `meals_with_macros.csv`.

```csv
meal_id,meal_text
1,"Ginger juice 100 ml Walnuts and dates 15 g Orange 60 g"
2,"Rice 150 g Steamed egg 100 g"
```

### Shanghai dataset

`scripts/prepare_shanghai_input.py` builds `data/raw/shanghai_meals.csv` from
`../data/shanghai_csv/meals.csv` (the output of `analysis/convert_shanghai.py`). It keeps only meals
with a recorded description, uses the English description as `meal_text`, and derives
`meal_id = {recording_id}_{YYYYMMDDTHHMM}`:

```bash
python scripts/prepare_shanghai_input.py
# Wrote 3467 meals (3123 unique texts) to data/raw/shanghai_meals.csv
```

## Running

Small test first:

```bash
python scripts/process_meals.py --input data/raw/shanghai_meals.csv --limit 10
```

Full dataset:

```bash
python scripts/process_meals.py --input data/raw/shanghai_meals.csv --output data/processed/meals_with_macros.csv
```

| Flag | Effect |
| --- | --- |
| `--limit N` | only the first N meals |
| `--batch-size N` | meals per LLM parse request (default 25) |
| `--force-reparse` | ignore cached meal parses |
| `--refresh-nutrition` | ignore cached nutrition (manual overrides still win) |
| `--no-resume` | both of the above |
| `--skip-llm-fallback` | foods without a confident database match stay `unresolved` |
| `--parser rule` | deterministic regex parser instead of the LLM (offline testing, no OpenAI key needed) |
| `--confidence-strategy` | `weight_weighted` (default), `calorie_weighted`, `min_item` |
| `--breakdown-output PATH` | where to write the per-food CSV |

At the end the script prints processed / successful / failed meals, unique foods, nutrition sources,
mean meal confidence, LLM requests, tokens and the estimated API cost.

## Caching and resume

- `cache/parsed_meals.json` is keyed by a hash of the meal text (identical texts are parsed once).
  Entries record the parser, model, prompt version and timestamp. It is written after every batch.
- `cache/nutrition_cache.json` is keyed by normalized food name. Database misses are cached too
  (`db_miss`), so reruns never query USDA again for them. A later run without
  `--skip-llm-fallback` sends them to the LLM.
- Writes are atomic (temp file then rename), so killing the process never corrupts a cache.
  Parse results are saved after every batch, database lookups every 5 foods, and LLM estimates
  after every batch.
- Failed parse batches and failed lookups are logged and **not** cached, so a rerun retries only them.
- Rerunning the same command after a crash resumes where it stopped.
- If the LLM fails for 3 batches in a row (for example a daily quota), the run stops calling it,
  still writes outputs for everything completed, and a later rerun continues.

### Save states and reference points

- **Snapshots**: every 5th checkpoint write (at most once every 2 minutes), and at the end of each
  stage, each cache is copied to `cache/snapshots/<cache>-<UTC timestamp>[-stage].json`. The newest
  30 per cache are kept (`SNAPSHOT_EVERY`, `SNAPSHOT_MIN_INTERVAL_S`, `SNAPSHOT_KEEP`).
- **Run lock**: `cache/.run.lock` stops a second run from writing the same caches at the same time.
  The OS releases it automatically if a run crashes or is killed.
- **Automatic recovery**: if a cache file is unreadable, it is renamed to `*.corrupt-<time>` and the
  newest valid snapshot is restored.
- **Manual restore**: `python scripts/restore_snapshot.py --list` shows every save state;
  `python scripts/restore_snapshot.py --restore cache/snapshots/<file>.json` restores one (the
  current file is backed up first).
- **Progress journal**: `cache/progress.jsonl` gets one line per event (run start, every parse
  batch, every 25 database lookups, every LLM fallback batch, run end) with the run id and counts.
- **Output history**: every run's CSVs and metadata are also copied to
  `data/processed/history/<run id>/`, so earlier results are never lost when outputs are rewritten.

## Manual corrections

`data/manual_nutrition_overrides.csv`:

```csv
normalized_food_name,calories_per_100g,carbs_per_100g,protein_per_100g,fat_per_100g,notes
coarse grain,116,25.6,3.1,0.6,dietitian estimate for hospital mixed-grain rice
```

Names are normalized the same way as parsed foods (see `normalized_food_name` in
`food_breakdown.csv`). Overrides take precedence over the cache, the database and the LLM, are
marked `source=manual_override`, `confidence=1.0`, and are never written by the pipeline.

## Outputs

`data/processed/meals_with_macros.csv`, one row per meal:

| column | meaning |
| --- | --- |
| `meal_id`, `meal_text` | from the input |
| `calories`, `carbs_g`, `protein_g`, `fat_g` | sum over resolved food items |
| `confidence` | meal confidence (see below) |
| `num_food_items`, `num_database_matches`, `num_llm_estimates` | item counts |
| `num_db_direct`, `num_db_approximate`, `num_manual_overrides`, `num_unresolved_items` | finer counts |
| `total_grams` | grams of the resolved items |
| `status` | `ok`, `partial` (some items unresolved), `no_nutrition`, `parse_failed`, `empty` |
| `parse_warnings` | validation warnings from parsing |
| passthrough columns | e.g. `patient_id`, `timestamp` |

`data/processed/food_breakdown.csv`, one row per food item: `meal_id`, `original_food_text`,
`normalized_food_name`, `amount`, `unit`, `estimated_grams`, `calories`, `carbs_g`, `protein_g`,
`fat_g`, `nutrition_source`, `nutrition_confidence`, `unit_conversion_confidence`, plus
`original_food_name`, `preparation`, `matched_food_name` (USDA description, or the rejected best
candidate for LLM estimates), `fdc_id`, `match_score`, `reasoning_summary` (LLM rationale),
`nutrition_flags`, `parser`, `parser_model`, `estimator_model`.

`data/processed/run_metadata.json` records the git commit, options, model, prompt versions,
thresholds, prices, token usage and summary of each run.

### Nutrition sources

| `nutrition_source` | meaning | confidence |
| --- | --- | --- |
| `manual_override` | from `manual_nutrition_overrides.csv` | 1.0 |
| `database_direct` | USDA food whose name matches the query (score >= 0.90) | 0.90-1.00 |
| `database_approximate` | USDA food containing every query word plus few extra qualifiers (score 0.80-0.89), or a mixed-dish name match | 0.70-0.89 |
| `llm_estimate` | LLM per-100 g estimate | <= 0.69; 0.30-0.49 for mixed dishes |
| `unresolved` | no usable value; macros are empty | 0 |

## Confidence scoring

- **Database match score**: share of query words found in the USDA name (squared), times how
  many of the USDA name's informative words come from the query. Penalties apply when the USDA head
  food differs ("Crackers, wonton" for "wonton"), when preparation words like *braised* or *fried*
  are missing, and when a dry or raw form is matched for staples weighed as served (rice, noodles,
  meat, eggs). The best of 50 candidates is used, never blindly the first one. Scores below 0.80
  go to the LLM fallback; the rejected candidate is kept in the breakdown for review. Database
  matches for mixed-dish names ("pork with bean") are capped at `database_approximate`, because
  US foods with the same words are often different recipes.
- **LLM estimates** are clamped to at most 0.69, and to 0.30-0.49 for mixed dishes (names containing
  a comma, *and* or *with*). The confidence is reduced, and flagged, when calories disagree with
  4C + 4P + 9F by more than 25% or macros sum to more than 100 g per 100 g.
- **Unit conversion**: mass units have confidence 1.0. Volumes use a keyword density table (water,
  tea, juice and soup 1.0 g/ml at 0.85-0.9; milk 1.03 at 0.9; oil 0.92 at 0.85). Unknown foods
  assume 1.0 g/ml at 0.5. Missing or unknown units give no grams (the item is kept, unresolved).
- **Meal confidence** (`weight_weighted`): `sum(nutrition_conf * unit_conf * grams) / sum(grams)` over
  resolved items, times the share of items that were resolved. Strategies live in
  `CONFIDENCE_STRATEGIES` in `src/macro_calculator.py`; add a function there to experiment.

## Tests

```bash
python -m pytest
```

Unit tests cover normalization, units, macro scaling and confidence, caching and override
precedence, match scoring, batching, retries and partial failures, and a mocked end-to-end run. No
network calls are made.

## Limitations

- **LLM-derived macros are estimates.** They reflect what a language model considers typical for a
  dish name, not a measurement or a validated food composition table. Treat `llm_estimate` rows as
  lower-quality labels; filter or weight them using `nutrition_source` and `confidence`.
- USDA FoodData Central describes mostly American foods. Many Shanghai dishes (coarse grain,
  Hangzhou cabbage, braised or stir-fried mixed dishes) have no faithful USDA equivalent and fall
  back to the LLM. A Chinese food composition table (e.g. China CDC's) would be a better provider;
  implement the `NutritionProvider` protocol in `src/nutrition_lookup.py` and pass it to the resolver.
- Recorded weights are assumed to be as served (cooked). Raw-vs-cooked mismatches change
  calories substantially for grains and meat.
- Cooking oil is only included when the LLM estimate or the matched USDA dish includes it. Plain
  vegetable matches (e.g. "Cabbage, raw") underestimate the fat in stir-fried vegetables.
- Mixed dishes with a single shared weight are never split into ingredients, so their values are
  dish-level estimates with low confidence.
- Volume-to-mass conversion for non-liquid foods is uncertain and flagged by low conversion confidence.
- Reproducibility: cached results make reruns identical. Regenerating with `--no-resume` can give
  slightly different LLM values even at temperature 0.
