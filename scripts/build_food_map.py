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

Needs USDA_API_KEY (read from backend/.env or the environment).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app import load_env_file                         # noqa: E402
from backend.food import usda                                 # noqa: E402
from backend.food.cache import DATA_DIR                       # noqa: E402
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="also build nutrient_cache.json")
    parser.add_argument("--labels", nargs="*", help="limit to these labels (default: all from the model config)")
    parser.add_argument("--out", default=str(DATA_DIR), help="output directory (default: backend/food/data)")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to sleep between USDA calls")
    args = parser.parse_args(argv)

    load_env_file(REPO / "backend" / ".env")
    out = Path(args.out)
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
