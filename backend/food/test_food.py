"""Tests for the food feature: parsing, scaling, portions, default selection, missing values.

No real network. USDA fixtures below reproduce the real response shapes captured from the
live API (flat search nutrients vs nested detail nutrients; FNDDS vs SR Legacy portions).
"""
import asyncio
import io
import unittest

import httpx
from fastapi import Depends, FastAPI
from PIL import Image

from . import identify as idf
from . import usda
from .cache import FoodCache
from .nutrition import (build_nutrition, estimate_kcal, extract_per_100g,
                        extract_portions, meal_text, scale_nutrients)
from .routes import build_food_router

# --- Fixtures shaped exactly like real USDA responses ----------------------------------

# /foods/search -> foods[].foodNutrients : flat, "nutrientNumber" + "value"
SEARCH_NUTRIENTS = [
    {"nutrientNumber": "203", "nutrientName": "Protein", "unitName": "G", "value": 1.84},
    {"nutrientNumber": "204", "nutrientName": "Total lipid (fat)", "unitName": "G", "value": 7.53},
    {"nutrientNumber": "205", "nutrientName": "Carbohydrate, by difference", "unitName": "G", "value": 32.38},
    {"nutrientNumber": "208", "nutrientName": "Energy", "unitName": "KCAL", "value": 210.0},
    {"nutrientNumber": "291", "nutrientName": "Fiber", "unitName": "G", "value": 1.2},
    {"nutrientNumber": "269", "nutrientName": "Sugars", "unitName": "G", "value": 5.0},
]

# /food/{id} -> foodNutrients : nested, "nutrient": {"number": ...} + "amount".
# Includes a group row (number 951, no amount) that must be ignored.
DETAIL_NUTRIENTS = [
    {"type": "FoodNutrient", "nutrient": {"number": "951", "name": "Proximates"}},  # group, no amount
    {"type": "FoodNutrient", "nutrient": {"number": "203", "name": "Protein", "unitName": "g"}, "amount": 1.84},
    {"type": "FoodNutrient", "nutrient": {"number": "204", "name": "Total lipid (fat)", "unitName": "g"}, "amount": 7.53},
    {"type": "FoodNutrient", "nutrient": {"number": "205", "name": "Carbohydrate", "unitName": "g"}, "amount": 32.38},
    {"type": "FoodNutrient", "nutrient": {"number": "208", "name": "Energy", "unitName": "kcal"}, "amount": 210.0},
]

# FNDDS portions: label in portionDescription; junk + odd entries present and out of order.
FNDDS_PORTIONS = {"foodPortions": [
    {"gramWeight": 10.5, "portionDescription": "1 surface inch", "sequenceNumber": 2,
     "measureUnit": {"name": "undetermined"}},
    {"gramWeight": 108, "portionDescription": "Quantity not specified", "sequenceNumber": 4,
     "measureUnit": {"name": "undetermined"}},
    {"gramWeight": 1188, "portionDescription": "1 pizza (12\" dia)", "sequenceNumber": 3,
     "measureUnit": {"name": "undetermined"}},
    {"gramWeight": 108, "portionDescription": "1 piece", "sequenceNumber": 1,
     "measureUnit": {"name": "undetermined"}},
]}

# SR Legacy portions: portionDescription is null; label built from amount + modifier.
SR_PORTIONS = {"foodPortions": [
    {"gramWeight": 140.0, "amount": 1.0, "modifier": "cup", "sequenceNumber": 1,
     "measureUnit": {"name": "undetermined"}, "portionDescription": None},
    {"gramWeight": 565.0, "amount": 1.0, "modifier": "serving", "sequenceNumber": 2,
     "measureUnit": {"name": "undetermined"}, "portionDescription": None},
]}


class ParsingTests(unittest.TestCase):
    def test_extract_nutrients_from_flat_search_shape(self):
        per = extract_per_100g(SEARCH_NUTRIENTS)
        self.assertEqual(per, {"kcal": 210.0, "protein_g": 1.84, "fat_g": 7.53,
                               "carbs_g": 32.38, "fiber_g": 1.2, "sugar_g": 5.0})

    def test_extract_nutrients_from_nested_detail_shape_skips_group_rows(self):
        per = extract_per_100g(DETAIL_NUTRIENTS)
        self.assertEqual(per["kcal"], 210.0)
        self.assertEqual(per["protein_g"], 1.84)
        self.assertIsNone(per["fiber_g"])  # not in this record
        self.assertIsNone(per["sugar_g"])

    def test_all_present(self):
        record = build_nutrition(1, "Dessert pizza", "Survey (FNDDS)", SEARCH_NUTRIENTS, FNDDS_PORTIONS)
        self.assertEqual(record["missing"], [])
        self.assertFalse(record["kcal_estimated"])
        self.assertEqual(record["per_100g"]["kcal"], 210.0)

    def test_kcal_missing_is_estimated_from_macros(self):
        nutrients = [n for n in SEARCH_NUTRIENTS if n["nutrientNumber"] != "208"]
        record = build_nutrition(1, "x", "SR Legacy", nutrients, None)
        self.assertTrue(record["kcal_estimated"])
        # 4*32.38 + 4*1.84 + 9*7.53 = 204.65 -> 205
        self.assertEqual(record["per_100g"]["kcal"], 205)
        self.assertNotIn("kcal", record["missing"])

    def test_kcal_missing_and_a_macro_missing_cannot_estimate(self):
        nutrients = [n for n in SEARCH_NUTRIENTS if n["nutrientNumber"] not in ("208", "205")]
        record = build_nutrition(1, "x", "SR Legacy", nutrients, None)
        self.assertFalse(record["kcal_estimated"])
        self.assertIsNone(record["per_100g"]["kcal"])
        self.assertIsNone(record["per_100g"]["carbs_g"])
        self.assertIn("kcal", record["missing"])
        self.assertIn("carbs_g", record["missing"])

    def test_a_macro_missing_is_null_never_zero(self):
        nutrients = [n for n in SEARCH_NUTRIENTS if n["nutrientNumber"] != "203"]  # drop protein
        record = build_nutrition(1, "x", "SR Legacy", nutrients, None)
        self.assertIsNone(record["per_100g"]["protein_g"])
        self.assertIn("protein_g", record["missing"])

    def test_no_portions_yields_empty_list_and_null_default(self):
        record = build_nutrition(1, "x", "SR Legacy", SEARCH_NUTRIENTS, {"foodPortions": []})
        self.assertEqual(record["portions"], [])
        self.assertIsNone(record["default_portion"])


class PortionTests(unittest.TestCase):
    def test_fndds_portions_use_description_drop_junk_and_order_by_sequence(self):
        portions = extract_portions(FNDDS_PORTIONS)
        labels = [p["label"] for p in portions]
        self.assertEqual(labels[0], "1 piece")              # sequenceNumber 1 first
        self.assertNotIn("Quantity not specified", labels)  # junk dropped
        self.assertIn("1 pizza (12\" dia)", labels)

    def test_sr_legacy_portions_build_label_from_amount_and_modifier(self):
        portions = extract_portions(SR_PORTIONS)
        self.assertEqual(portions[0], {"label": "1 cup", "grams": 140.0})
        self.assertEqual(portions[1], {"label": "1 serving", "grams": 565.0})

    def test_default_portion_is_first_by_sequence(self):
        record = build_nutrition(1, "Dessert pizza", "Survey (FNDDS)", SEARCH_NUTRIENTS, FNDDS_PORTIONS)
        self.assertEqual(record["default_portion"]["label"], "1 piece")
        self.assertEqual(record["default_portion"]["grams"], 108)


class ScaleTests(unittest.TestCase):
    PER_100G = {"kcal": 210.0, "protein_g": 1.84, "fat_g": 7.53,
                "carbs_g": 32.38, "fiber_g": None, "sugar_g": None}

    def test_normal_scaling_rounds_kcal_to_whole_and_grams_to_one_dp(self):
        scaled = scale_nutrients(self.PER_100G, 108)
        self.assertEqual(scaled["kcal"], round(210.0 * 1.08))     # 227
        self.assertEqual(scaled["carbs_g"], round(32.38 * 1.08, 1))  # 35.0
        self.assertEqual(scaled["protein_g"], round(1.84 * 1.08, 1))

    def test_zero_grams_is_all_zero(self):
        scaled = scale_nutrients(self.PER_100G, 0)
        self.assertEqual(scaled["kcal"], 0)
        self.assertEqual(scaled["carbs_g"], 0.0)

    def test_null_passes_through_unchanged(self):
        scaled = scale_nutrients(self.PER_100G, 150)
        self.assertIsNone(scaled["fiber_g"])
        self.assertIsNone(scaled["sugar_g"])

    def test_estimate_kcal_needs_all_three_macros(self):
        self.assertIsNone(estimate_kcal({"carbs_g": 10, "protein_g": None, "fat_g": 1}))
        self.assertEqual(estimate_kcal({"carbs_g": 10, "protein_g": 2, "fat_g": 1}), round(4*10 + 4*2 + 9*1))


class MealTextTests(unittest.TestCase):
    def test_meal_text_matches_forecaster_format(self):
        self.assertEqual(meal_text("Dessert pizza", 108), "Dessert pizza 108 g")
        self.assertEqual(meal_text("  Rice ", 150.0), "Rice 150 g")


# --- Image identification (no model download; a fake classifier stands in) --------------

def png_bytes(color=(200, 120, 40), size=(32, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_classifier(top_score):
    """A classifier callable like the transformers pipeline: image -> scored label list."""
    def classify(image):
        return [{"label": "pizza", "score": top_score},
                {"label": "french_fries", "score": (1 - top_score) * 0.6},
                {"label": "hamburger", "score": (1 - top_score) * 0.3},
                {"label": "hot_dog", "score": (1 - top_score) * 0.07},
                {"label": "tacos", "score": (1 - top_score) * 0.02},
                {"label": "churros", "score": (1 - top_score) * 0.01}]
    return classify


class IdentifyUnitTests(unittest.TestCase):
    def test_top5_display_names_and_confidence_range(self):
        result = idf.identify(png_bytes(), "image/png", threshold=0.6, classifier=fake_classifier(0.9))
        self.assertEqual(len(result["predictions"]), 5)  # top-5 only
        first = result["predictions"][0]
        self.assertEqual(first["label"], "pizza")
        self.assertEqual(first["display_name"], "Pizza")
        self.assertTrue(0 <= first["confidence"] <= 1)
        self.assertEqual(result["predictions"][1]["display_name"], "French Fries")

    def test_needs_confirmation_below_threshold(self):
        low = idf.identify(png_bytes(), "image/png", threshold=0.6, classifier=fake_classifier(0.4))
        self.assertTrue(low["needs_confirmation"])
        high = idf.identify(png_bytes(), "image/png", threshold=0.6, classifier=fake_classifier(0.8))
        self.assertFalse(high["needs_confirmation"])

    def test_rejects_non_image_bytes(self):
        with self.assertRaises(idf.ImageValidationError):
            idf.validate_image(b"not an image", "image/png")

    def test_rejects_oversize(self):
        with self.assertRaises(idf.ImageValidationError):
            idf.validate_image(b"x" * (idf.MAX_UPLOAD_BYTES + 1), "image/png")

    def test_rejects_wrong_content_type(self):
        with self.assertRaises(idf.ImageValidationError):
            idf.validate_image(png_bytes(), "application/pdf")


# --- Endpoint tests via a minimal app (dependency override, no auth dance) --------------

class ApiClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
                                         base_url="http://test") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())


SEARCH_FOODS = [{"fdcId": 2708008, "description": "Dessert pizza", "dataType": "Survey (FNDDS)",
                 "foodNutrients": SEARCH_NUTRIENTS}]


class RouteTests(unittest.TestCase):
    def build(self, cache=None, threshold=0.6):
        app = FastAPI()
        app.include_router(build_food_router(lambda: "test-user", cache=cache or FoodCache(),
                                             threshold=threshold))
        return ApiClient(app)

    def preloaded_cache(self):
        cache = FoodCache(data_dir="/nonexistent-so-empty")
        cache.food_map = {"pizza": {"fdcId": 2708008, "description": "Dessert pizza", "dataType": "Survey (FNDDS)"}}
        cache.nutrients = {"2708008": build_nutrition(2708008, "Dessert pizza", "Survey (FNDDS)",
                                                      SEARCH_NUTRIENTS, FNDDS_PORTIONS)}
        return cache

    def test_cache_first_lookup_by_id_makes_no_network_call(self):
        cache = self.preloaded_cache()
        client = self.build(cache)

        def boom(*a, **k):
            raise AssertionError("USDA was called despite a cache hit")
        original_detail, original_search = usda.food_detail, usda.search_foods
        usda.food_detail, usda.search_foods = boom, boom
        try:
            r = client.request("GET", "/food/nutrients/2708008")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["fdcId"], 2708008)
            # Mapped-label lookup is also cache-first (no network).
            r2 = client.request("GET", "/food/nutrients", params={"label": "pizza"})
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(r2.json()["description"], "Dessert pizza")
        finally:
            usda.food_detail, usda.search_foods = original_detail, original_search

    def test_search_shape_and_caching(self):
        client = self.build()
        calls = {"n": 0}

        def fake_search(query, page_size=10, data_types=None, client=None):
            calls["n"] += 1
            return SEARCH_FOODS
        original = usda.search_foods
        usda.search_foods = fake_search
        try:
            r = client.request("GET", "/food/search", params={"q": "pizza"})
            self.assertEqual(r.status_code, 200)
            results = r.json()["results"]
            self.assertEqual(results[0]["fdcId"], 2708008)
            self.assertEqual(results[0]["portions"], [])  # portions come on selection
            self.assertIn("per_100g", results[0])
            # Second identical query is served from the in-memory cache.
            client.request("GET", "/food/search", params={"q": "pizza"})
            self.assertEqual(calls["n"], 1)
        finally:
            usda.search_foods = original

    def test_identify_endpoint_with_injected_model(self):
        client = self.build(threshold=0.6)
        original = idf.get_classifier
        idf.get_classifier = lambda: fake_classifier(0.95)
        try:
            r = client.request("POST", "/food/identify",
                               files={"image": ("meal.png", png_bytes(), "image/png")})
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["predictions"][0]["label"], "pizza")
            self.assertFalse(body["needs_confirmation"])
        finally:
            idf.get_classifier = original

    def test_identify_rejects_bad_image_with_400(self):
        client = self.build()
        r = client.request("POST", "/food/identify",
                           files={"image": ("x.png", b"nope", "image/png")})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
