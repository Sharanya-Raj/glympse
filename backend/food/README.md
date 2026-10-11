# Photo / typed food → USDA nutrients & standard portions

A self-contained feature: identify a food (from a **photo** or by **typing** it), look up its
nutrients and USDA's **standard portion** from FoodData Central, show the user that portion so
they can compare it to what's on their plate, and feed the food to the existing glucose
forecaster as the same `"<food> <grams> g"` text it already reads.

Photo identification uses a local image classifier
(`nateraw/food`, a Vision Transformer fine-tuned on Food-101, 101 classes). Nutrition comes
only from USDA FoodData Central. Meal dictation uses ElevenLabs transcription and Gemini
to interpret foods and select USDA matches; the same pipeline is available in the CLI.

## How the pieces fit

| File | Role |
|------|------|
| `usda.py` | Thin FDC client. **POST** `/foods/search` (GET is rejected by USDA's nginx on some queries), **GET** `/food/{fdcId}`. Key from `USDA_API_KEY` (or `FDC_API_KEY`). |
| `nutrition.py` | Parse FDC records into one stable shape; `scale_nutrients`, portion extraction, kcal estimation. Handles both the flat *search* nutrient shape and the nested *detail* shape. |
| `identify.py` | Lazy-loaded image classifier + upload validation. torch/transformers are optional at import. |
| `cache.py` | Loads the committed offline data (`data/food_map.json`, `data/nutrient_cache.json`) and a small in-memory cache for live lookups. |
| `routes.py` | FastAPI router mounted by `backend.app.create_app`. |
| `meal.py` | Shared voice meal pipeline: ElevenLabs transcription, Gemini interpretation, USDA matching; optional CLI microphone recording. |
| `data/` | Committed offline data so the demo works with no network call. |

## Endpoints (authenticated, no `/api` prefix — matching the rest of the app)

- `POST /food/identify` — multipart `image` → `{ predictions: [{label, display_name, confidence}], needs_confirmation }`.
  Validates jpg/png/webp, ≤ 10 MB, opened with Pillow and converted to RGB. `needs_confirmation`
  is true when the top confidence is below `FOOD_CONFIDENCE_THRESHOLD` (default 0.6).
- `GET /food/nutrients?label=<label>` — nutrition for a classifier label (cache-first; live USDA fallback when unmapped).
- `GET /food/nutrients/{fdcId}` — nutrition for a specific FDC food.
- `GET /food/search?q=<text>` — up to 10 USDA matches (powers "none of these, search"). Portions arrive when one is selected.
- `POST /food/dictate` — multipart `audio` → transcript, warnings, and foods with USDA matches and nutrition.
  Accepts WebM, Ogg, MP4/M4A, WAV, or MP3 up to 10 MB. Requires authentication.

## Dictate a meal in the frontend

In **What if → Foods and portions**, click **Dictate my meal**, allow microphone access,
describe foods and amounts, and click **Stop and transcribe**. Recording stops automatically
after two minutes. Review the transcript and USDA matches, then click **Use dictated meal**
to replace the food list. Medication choices stay unchanged. Edit foods or search USDA to
correct a match; quantities that cannot be converted to grams require a portion before prediction.
Stated amounts are used directly without a portion reminder. If an amount is omitted, the USDA
standard serving is filled in and marked for a double-check. When USDA has no standard serving,
the app explicitly assumes 100 g and asks the user to double-check that fallback.
Counts such as "three eggs" are matched to a compatible USDA reference (for example "1 large")
and multiplied by the spoken quantity. The food list shows an editable **Number of portions**
and the total grams. One row represents the full amount of that food, not one physical item.
Gemini chooses only among USDA-provided portion labels; gram weights and scaled nutrients
come from USDA. If no compatible portion exists, the spoken count remains visible for review.
Cancel discards the recording or pending result. Leaving the meal screen also stops recording.

Use HTTPS or localhost for microphone access. Audio is sent to ElevenLabs Scribe v2;
Gemini interprets the transcript and selects from USDA candidates. Only USDA supplies nutrients.
The app holds audio in memory and does not save recordings to disk. Provider calls already
started on the server may finish even after the browser cancels.

Set these server-only keys in `backend/.env` and restart the backend:

```ini
ELEVEN_LABS_API_KEY=your-elevenlabs-key
GEMINI_API_KEY=your-gemini-key
USDA_API_KEY=your-usda-key
# Optional: GEMINI_MODEL=gemini-3.8-flash
```

The root `.env` is also loaded as a fallback. Process environment variables take precedence,
followed by `backend/.env`, then root `.env`. Gemini also accepts `GOOGLE_API_KEY` or
`GOOGLE_API_AI`, so existing local credentials can be reused.

Install `backend/requirements.txt` for the Gemini SDK. Browser recording does not need
`sounddevice`, torch, or transformers. Missing keys and provider errors appear in the dictation
panel; existing food entries are preserved until you choose **Use dictated meal**.

Provider references: [ElevenLabs speech-to-text](https://elevenlabs.io/docs/api-reference/speech-to-text/convert),
[Gemini structured output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).

Nutrition shape:

```json
{ "fdcId": 2708008, "description": "Dessert pizza", "source": "Survey (FNDDS)",
  "per_100g": { "kcal": 204, "protein_g": 1.84, "fat_g": 7.53, "carbs_g": 32.4, "fiber_g": 1.7, "sugar_g": 15.8 },
  "portions": [ { "label": "1 piece", "grams": 108 }, { "label": "1 pizza (12\" dia)", "grams": 1188 } ],
  "default_portion": { "label": "1 piece", "grams": 108 },
  "missing": [], "kcal_estimated": false }
```

`fiber_g`/`sugar_g` are fetched and cached but not shown in the UI yet. If kcal is missing it is
estimated as `4*carbs + 4*protein + 9*fat` and `kcal_estimated` is set; if a macro is missing it
is returned as `null` (never 0) and listed in `missing`. The glucose model consumes the food
**name + grams** (not the macros) — macros are shown for comparison only.

## Setup

1. **USDA key** — get one free at <https://fdc.nal.usda.gov/api-key-signup.html> and put it in
   `backend/.env` as `USDA_API_KEY=...` (gitignored; see `backend/.env.example`). The shared
   `DEMO_KEY` is limited to 30 requests/hour, a real key to 1,000/hour — hence the caching.

2. **Offline food data is already committed** under `data/`. To rebuild it (e.g. after editing
   mappings by hand):

   ```bash
   python scripts/build_food_map.py            # writes data/food_map_review.csv + data/food_map.json
   python scripts/build_food_map.py --fetch     # also writes data/nutrient_cache.json
   ```

   `food_map_review.csv` lists the top-5 USDA candidates per label for **manual review** —
   the top candidate is used by default and some are imperfect (e.g. `pizza → "Dessert pizza"`,
   `spaghetti_bolognese → "Spaghetti sauce"`). Edit `food_map.json` by hand and re-run `--fetch`
   to correct them. At runtime the backend uses these files with no network call, falling back
   to a live USDA lookup only when a label is unmapped.

3. **Photo identification (optional, heavy)** — install torch + transformers to enable it:

   ```bash
   python -m pip install -r backend/food/requirements.txt
   ```

   The first `POST /food/identify` downloads the `nateraw/food` model (~500 MB). Without these
   packages the endpoint returns a clean 503 and the **typed-food + USDA flow still works**.

## Voice meal mapping (CLI)

Describe a meal out loud (optionally with a photo) and get a meal-specific USDA mapping:

```bash
python scripts/build_food_map.py --record --image meal.jpg   # Enter to start, Enter to stop (max 2 min)
python scripts/build_food_map.py --record
python scripts/build_food_map.py --audio meal.wav --image meal.jpg --fetch
```

Needs `ELEVEN_LABS_API_KEY` and `GEMINI_API_KEY` (optional `GEMINI_MODEL`, default
`gemini-3.8-flash`) in `backend/.env`, plus `python -m pip install sounddevice google-genai`.
ElevenLabs Scribe transcribes the speech; Gemini turns the transcript (and photo) into foods with
stated quantities, then picks among the top 5 USDA search results for foods that are not already
in `food_map.json`. Speech wins over the photo: corrections and exclusions are applied, foods
seen only in the photo are never added, and disagreements become warnings. Unstated quantities
stay `null`.

The result goes to `outputs/meals/<timestamp>-<id>.json` (or `--meal-out`): transcript,
`image_used`, warnings and ordered `items` with `match_status` (`mapped` from `food_map.json`,
`matched` live, `unmatched`, or `error`) and a nullable `{fdcId, description, dataType}` match.
`--fetch` embeds per-100 g nutrition and a serving from the stated weight or compatible USDA
portion (e.g. "2 cups"). Omitted amounts use a standard serving (100 g if unavailable), marked
with `serving.assumed: true`; explicit amounts have `assumed: false`. Meal mode never modifies `food_map.json`,
`food_map_review.csv` or `nutrient_cache.json`.

## Tests

```bash
python -m pytest backend/food/test_food.py backend/food/test_meal.py
```

No real network: USDA responses are mocked with fixtures that reproduce the real shapes, and the
classifier is replaced by a fake. Covers `scale_nutrients` (normal/zero/null), parsing (all
present, kcal missing, a macro missing, no portions), `default_portion` selection, the
`needs_confirmation` threshold, upload validation, and that a cache-first lookup makes no network call.

## Known limitations (also surfaced in the UI)

- **Only 101 food classes**, mostly restaurant dishes; plain staples may be missing — that's why
  the typeable box and "none of these, search" fallback exist. 7 Food-101 labels have no clean
  USDA auto-match and rely on the search box.
- **One label per photo.** A photo can't tell portion size, so we show USDA's standard portion
  for the user to compare against their plate.
- The glucose forecaster keys off the food **name** (category) and **grams**, so a dish the
  category regex doesn't recognize is treated as "other" food of that weight.
- Nutrition shown is an **estimate for wellness, not medical advice.**
