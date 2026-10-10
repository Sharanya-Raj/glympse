# Photo / typed food → USDA nutrients & standard portions

A self-contained feature: identify a food (from a **photo** or by **typing** it), look up its
nutrients and USDA's **standard portion** from FoodData Central, show the user that portion so
they can compare it to what's on their plate, and feed the food to the existing glucose
forecaster as the same `"<food> <grams> g"` text it already reads.

**No LLM or generative-AI calls anywhere.** Identification uses a local image classifier
(`nateraw/food`, a Vision Transformer fine-tuned on Food-101, 101 classes). Nutrition comes
only from USDA FoodData Central.

## How the pieces fit

| File | Role |
|------|------|
| `usda.py` | Thin FDC client. **POST** `/foods/search` (GET is rejected by USDA's nginx on some queries), **GET** `/food/{fdcId}`. Key from `USDA_API_KEY` (or `FDC_API_KEY`). |
| `nutrition.py` | Parse FDC records into one stable shape; `scale_nutrients`, portion extraction, kcal estimation. Handles both the flat *search* nutrient shape and the nested *detail* shape. |
| `identify.py` | Lazy-loaded image classifier + upload validation. torch/transformers are optional at import. |
| `cache.py` | Loads the committed offline data (`data/food_map.json`, `data/nutrient_cache.json`) and a small in-memory cache for live lookups. |
| `routes.py` | FastAPI router mounted by `backend.app.create_app`. |
| `data/` | Committed offline data so the demo works with no network call. |

## Endpoints (authenticated, no `/api` prefix — matching the rest of the app)

- `POST /food/identify` — multipart `image` → `{ predictions: [{label, display_name, confidence}], needs_confirmation }`.
  Validates jpg/png/webp, ≤ 10 MB, opened with Pillow and converted to RGB. `needs_confirmation`
  is true when the top confidence is below `FOOD_CONFIDENCE_THRESHOLD` (default 0.6).
- `GET /food/nutrients?label=<label>` — nutrition for a classifier label (cache-first; live USDA fallback when unmapped).
- `GET /food/nutrients/{fdcId}` — nutrition for a specific FDC food.
- `GET /food/search?q=<text>` — up to 10 USDA matches (powers "none of these, search"). Portions arrive when one is selected.

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

## Tests

```bash
python -m pytest backend/food/test_food.py
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
