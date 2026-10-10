from src.models import SOURCE_DB_APPROX, SOURCE_DB_DIRECT, SOURCE_DB_REJECTED
from src.nutrition_lookup import USDAProvider, extract_macros, match_score, score_to_source_confidence


def source_for(query, candidate, settings):
    return score_to_source_confidence(match_score(query, candidate).score, settings)


def test_simple_whole_food_is_direct(settings):
    source, conf = source_for("orange", "Oranges, raw, all commercial varieties", settings)
    assert source == SOURCE_DB_DIRECT and conf >= 0.9


def test_prepared_dish_vs_raw_ingredient_is_rejected(settings):
    source, _ = source_for("braised duck wing in brown sauce", "Duck, domesticated, meat only, raw", settings)
    assert source == SOURCE_DB_REJECTED


def test_different_head_food_is_rejected(settings):
    source, _ = source_for("wonton", "Crackers, wonton", settings)
    assert source == SOURCE_DB_REJECTED


def test_extra_qualifiers_give_approximate(settings):
    source, conf = source_for("leaf lettuce", "Lettuce, leaf, green, raw", settings)
    assert source == SOURCE_DB_APPROX and 0.7 <= conf < 0.9


def test_single_word_with_foreign_qualifier_is_rejected(settings):
    source, _ = source_for("bread", "Bread, cheese", settings)
    assert source == SOURCE_DB_REJECTED


def test_mixed_dish_database_match_is_at_most_approximate(settings):
    nutrients = _food("Pork and beans")
    session = _Session([_Resp(200, {"foods": [nutrients]})])
    provider = USDAProvider(settings, session=session, sleep=lambda s: None)
    provider.min_interval = 0
    result = provider.search("minced pork with bean")
    assert result.match_score == 1.0
    assert result.source == SOURCE_DB_APPROX and result.confidence <= 0.89
    assert "mixed_dish_database_match" in result.flags


def test_missing_ingredient_is_rejected(settings):
    source, _ = source_for("mushroom and bamboo shoots", "Bamboo shoots, raw", settings)
    assert source == SOURCE_DB_REJECTED


def test_cutting_words_are_ignored():
    assert match_score("sliced lotus root", "Lotus root, raw").score == 1.0


def test_cooked_staple_prefers_cooked_form():
    cooked = match_score("rice", "Rice, cooked, NFS").score
    raw = match_score("rice", "Rice, white, long-grain, regular, raw, enriched").score
    dry = match_score("rice", "Rice flour, dry").score
    assert cooked > raw and cooked > dry


def test_preparation_mismatch_is_penalized():
    assert match_score("fried egg", "Egg, whole, cooked, fried").score > match_score("fried egg", "Egg, whole, raw").score


def _food(desc, kcal=50, data_type="SR Legacy", **extra):
    nutrients = [
        {"nutrientId": 1008, "unitName": "KCAL", "value": kcal},
        {"nutrientId": 1003, "unitName": "G", "value": 1},
        {"nutrientId": 1004, "unitName": "G", "value": 0.2},
        {"nutrientId": 1005, "unitName": "G", "value": 12},
    ]
    return {"fdcId": hash(desc) % 10**6, "description": desc, "dataType": data_type, "foodNutrients": nutrients, **extra}


def test_ranking_does_not_blindly_take_first_result(settings):
    provider = USDAProvider(settings, session=object())
    ranked = provider.rank_candidates("orange", [_food("Orange chicken", 250), _food("Oranges, raw, all commercial varieties")])
    assert ranked[0][1]["description"] == "Oranges, raw, all commercial varieties"


def test_candidates_without_full_macros_are_skipped(settings):
    provider = USDAProvider(settings, session=object())
    incomplete = {"description": "Orange, raw", "foodNutrients": [{"nutrientId": 1008, "unitName": "KCAL", "value": 47}]}
    assert provider.rank_candidates("orange", [incomplete]) == []


def test_extract_macros_falls_back_to_atwater_energy_and_alternate_ids():
    food = {"foodNutrients": [
        {"nutrientId": 1062, "unitName": "kJ", "value": 200},
        {"nutrientId": 2047, "unitName": "KCAL", "value": 48},
        {"nutrientId": 1003, "unitName": "G", "value": 0.9},
        {"nutrientId": 1085, "unitName": "G", "value": 0.1},
        {"nutrientId": 1050, "unitName": "G", "value": 11.8},
    ]}
    assert extract_macros(food) == {
        "calories_per_100g": 48, "protein_per_100g": 0.9, "fat_per_100g": 0.1, "carbs_per_100g": 11.8,
    }


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_usda_search_retries_rate_limit(settings):
    session = _Session([_Resp(429), _Resp(200, {"foods": [_food("Orange, raw")]})])
    provider = USDAProvider(settings, session=session, sleep=lambda s: None)
    provider.min_interval = 0
    result = provider.search("orange")
    assert session.calls == 2
    assert result.source == SOURCE_DB_DIRECT and result.matched_food_name == "Orange, raw"
