"""Pre-map the classifier's labels to USDA foods so the demo works offline.

Reads the model's own label list from nateraw/food's config.json (id2label), searches USDA
FoodData Central for each label, and writes:

  backend/food/data/food_map_review.csv  top-5 USDA candidates per label, for manual review
  backend/food/data/food_map.json        label -> {fdcId, description, dataType} (top candidate)

With --fetch it also pulls nutrients + portions for every mapped fdcId into:

  backend/food/data/nutrient_cache.json  str(fdcId) -> nutrition record (per-100 g + portions)

At runtime the backend uses these files with no network call, falling back to a live USDA
lookup only when a label is missing. Commit the generated files.

Usage (from the repo root):
  python scripts/build_food_map.py                # map labels, write review CSV + food_map.json
  python scripts/build_food_map.py --fetch        # also build nutrient_cache.json
  python scripts/build_food_map.py --labels apple_pie pizza   # limit to a few labels (testing)

Meal mode maps one spoken meal (plus an optional photo) to USDA foods and writes a meal JSON
(default outputs/meals/<timestamp>-<id>.json). It never modifies the three shared files above:
  python scripts/build_food_map.py --record --image meal.jpg  # speak about a meal, with a photo
  python scripts/build_food_map.py --record                   # describe a meal without a photo
  python scripts/build_food_map.py --audio meal.wav --fetch   # existing recording, plus nutrition

Needs USDA_API_KEY (read from backend/.env or the environment). Meal mode also needs
ELEVEN_LABS_API_KEY and GEMINI_API_KEY (optional GEMINI_MODEL), plus
  python -m pip install sounddevice google-genai
"""
import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app import load_env_file                         # noqa: E402
from backend.food import meal, usda                           # noqa: E402
from backend.food.cache import DATA_DIR, FOOD_MAP_FILE, NUTRIENT_CACHE_FILE, load_json  # noqa: E402
from backend.food.nutrition import build_nutrition            # noqa: E402

CONFIG_URL = "https://huggingface.co/nateraw/food/resolve/main/config.json"


def model_labels():
    """The classifier's labels, read from its published config (id2label) rather than assumed."""
    response = httpx.get(CONFIG_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    id2label = response.json()["id2label"]
    # Keys are string indices "0".."100"; keep model order.
    return [id2label[str(i)] for i in range(len(id2label))]


def display_name(label):
    return label.replace("_", " ").strip()


def candidates(label, client):
    """Top USDA candidates for one label (search by the human-readable name)."""
    foods = usda.search_foods(display_name(label), page_size=5, client=client)
    return [{"fdcId": f.get("fdcId"), "description": f.get("description"),
             "dataType": f.get("dataType"), "score": f.get("score")} for f in foods]


def build_map(labels, client, pause):
    review_rows, food_map, missing = [], {}, []
    for i, label in enumerate(labels, 1):
        try:
            found = candidates(label, client)
        except usda.UsdaError as exc:
            print(f"  [{i}/{len(labels)}] {label}: USDA error: {exc}", file=sys.stderr)
            found = []
        for rank, c in enumerate(found, 1):
            review_rows.append({"label": label, "display_name": display_name(label), "rank": rank,
                                "fdcId": c["fdcId"], "description": c["description"],
                                "dataType": c["dataType"], "score": c["score"]})
        if found:
            top = found[0]
            food_map[label] = {"fdcId": top["fdcId"], "description": top["description"],
                               "dataType": top["dataType"]}
            print(f"  [{i}/{len(labels)}] {label} -> {top['fdcId']} {top['description']} ({top['dataType']})")
        else:
            missing.append(label)
            review_rows.append({"label": label, "display_name": display_name(label), "rank": "",
                                "fdcId": "", "description": "NO MATCH", "dataType": "", "score": ""})
            print(f"  [{i}/{len(labels)}] {label} -> NO MATCH", file=sys.stderr)
        if pause:
            time.sleep(pause)
    return review_rows, food_map, missing


def fetch_nutrients(food_map, client, pause):
    cache = {}
    items = list(food_map.items())
    for i, (label, entry) in enumerate(items, 1):
        fdc_id = entry["fdcId"]
        if str(fdc_id) in cache:
            continue
        try:
            detail = usda.food_detail(fdc_id, client=client)
        except usda.UsdaError as exc:
            print(f"  [{i}/{len(items)}] {label} ({fdc_id}): USDA error: {exc}", file=sys.stderr)
            continue
        record = build_nutrition(detail.get("fdcId", fdc_id), detail.get("description", entry["description"]),
                                 detail.get("dataType", entry["dataType"]), detail.get("foodNutrients"), detail)
        cache[str(record["fdcId"])] = record
        flags = (" estimated-kcal" if record["kcal_estimated"] else "") + \
                (f" missing={record['missing']}" if record["missing"] else "") + \
                (" NO-PORTIONS" if not record["portions"] else f" portions={len(record['portions'])}")
        print(f"  [{i}/{len(items)}] {label} ({record['fdcId']}):{flags}")
        if pause:
            time.sleep(pause)
    return cache


SHARED_FILES = (FOOD_MAP_FILE, "food_map_review.csv", NUTRIENT_CACHE_FILE)


def record_meal_audio():
    """Enter to start, Enter to stop (or MAX_RECORD_SECONDS); returns WAV bytes kept in memory."""
    if not sys.stdin.isatty():
        raise meal.MealError("--record needs an interactive terminal; use --audio PATH instead.")
    input("Press Enter to start recording (Ctrl+C cancels) ...")
    stop = threading.Event()

    def wait_for_enter():
        try:
            input()
        except (EOFError, OSError):
            pass
        stop.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    print(f"Recording ... press Enter to stop (max {meal.MAX_RECORD_SECONDS // 60} min).")
    audio, seconds = meal.record_wav(stop)
    print(f"Recorded {seconds:.1f} s.")
    return audio


def print_meal_summary(document):
    photo = "photo used" if document["image_used"] else "no photo"
    print(f"\nMeal: {len(document['items'])} food(s), {photo}")
    for i, item in enumerate(document["items"], 1):
        amount = " ".join(f"{part:g}" if isinstance(part, float) else str(part)
                          for part in (item["quantity"], item["unit"]) if part is not None)
        details = ", ".join(part for part in (amount, item["preparation"]) if part)
        name = f"{item['name']} ({details})" if details else item["name"]
        match = item["match"]
        target = (f"{match['fdcId']} {match['description']} ({match['dataType']})" if match
                  else f"{item['match_status'].upper()} (search: '{item['search_terms']}')")
        print(f"  {i}. {name} -> {target}")
        serving = (item.get("nutrition") or {}).get("serving")
        if serving:
            nutrients = serving["nutrients"]
            print(f"     {serving['grams']:g} g ({serving['basis']}): {nutrients['kcal']} kcal, "
                  f"{nutrients['carbs_g']} g carbs, {nutrients['protein_g']} g protein, {nutrients['fat_g']} g fat")
        for warning in item["warnings"]:
            print(f"     warning: {warning}")
    for warning in document["warnings"]:
        print(f"Warning: {warning}")


def run_meal(args):
    load_env_file(REPO / "backend" / ".env")
    load_env_file(REPO / ".env")
    out_path = Path(args.meal_out) if args.meal_out else meal.default_meal_path(REPO)
    if out_path.resolve() in {(DATA_DIR / name).resolve() for name in SHARED_FILES}:
        raise meal.MealError(f"--meal-out must not overwrite shared food data ({out_path}).")
    stt_key = meal.elevenlabs_key()
    gemini = meal.Gemini(meal.gemini_key())
    image = meal.prepare_image(args.image) if args.image else None

    if args.record:
        audio, filename, content_type = record_meal_audio(), "meal.wav", "audio/wav"
    else:
        audio, filename, content_type = meal.read_audio_file(args.audio)
    print("Transcribing ...")
    transcript = meal.transcribe(audio, filename, content_type, stt_key)
    print(f'Transcript: "{transcript}"')

    print("Understanding meal ...")
    interpretation = meal.interpret_meal(transcript, gemini, image)

    print("Matching foods ...")
    food_map = load_json(DATA_DIR / FOOD_MAP_FILE, {})
    with httpx.Client(timeout=usda.TIMEOUT) as client:
        def search(term):
            try:
                return candidates(term, client)
            finally:
                if args.pause:
                    time.sleep(args.pause)

        items = meal.match_items(interpretation.items, food_map, search, gemini)
        if args.fetch:
            print("Fetching nutrients ...")
            records = load_json(DATA_DIR / NUTRIENT_CACHE_FILE, {})
            missing = {f"{i}:{item['name']}": item["match"] for i, item in enumerate(items, 1)
                       if item["match"] and str(item["match"]["fdcId"]) not in records}
            if missing:
                records = {**records, **fetch_nutrients(missing, client, args.pause)}
            meal.attach_nutrition(items, records, gemini=gemini)

    document = meal.build_meal(transcript, image is not None, interpretation, items, gemini.model, args.fetch)
    meal.write_json_atomic(out_path, document)
    print_meal_summary(document)
    print(f"\nWrote {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true",
                        help="also build nutrient_cache.json (meal mode: embed nutrition in the meal file)")
    parser.add_argument("--labels", nargs="*", help="limit to these labels (default: all from the model config)")
    parser.add_argument("--out", help="output directory (default: backend/food/data)")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between USDA calls")
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument("--record", action="store_true", help="meal mode: describe a meal into the microphone")
    voice.add_argument("--audio", metavar="PATH", help="meal mode: transcribe an existing recording")
    parser.add_argument("--image", metavar="PATH", help="meal mode: optional JPG/PNG/WebP photo of the meal")
    parser.add_argument("--meal-out", metavar="PATH",
                        help="meal mode output file (default: outputs/meals/<timestamp>-<id>.json)")
    args = parser.parse_args(argv)

    if args.record or args.audio is not None:
        if args.labels is not None or args.out is not None:
            parser.error("--labels and --out cannot be combined with --record/--audio")
        try:
            run_meal(args)
        except meal.MealError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        except KeyboardInterrupt:
            print("\nCancelled; nothing was written.", file=sys.stderr)
            raise SystemExit(130)
        return
    if args.image is not None or args.meal_out is not None:
        parser.error("--image and --meal-out need a voice input (--record or --audio PATH)")

    load_env_file(REPO / "backend" / ".env")
    out = Path(args.out or DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    labels = args.labels or model_labels()
    print(f"Mapping {len(labels)} labels to USDA foods ...")
    with httpx.Client(timeout=usda.TIMEOUT) as client:
        review_rows, food_map, missing = build_map(labels, client, args.pause)

        review_path = out / "food_map_review.csv"
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "display_name", "rank",
                                                        "fdcId", "description", "dataType", "score"])
            writer.writeheader()
            writer.writerows(review_rows)

        map_path = out / "food_map.json"
        map_path.write_text(json.dumps(food_map, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {map_path} ({len(food_map)} mapped, {len(missing)} unmatched)")
        print(f"Wrote {review_path} (review by hand; re-run after editing food_map.json if needed)")
        if missing:
            print("Unmatched labels (UI search box covers these at runtime):", ", ".join(missing))

        if args.fetch:
            print("\nFetching nutrients + portions for mapped foods ...")
            cache = fetch_nutrients(food_map, client, args.pause)
            cache_path = out / "nutrient_cache.json"
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Wrote {cache_path} ({len(cache)} foods cached)")


if __name__ == "__main__":
    main()
