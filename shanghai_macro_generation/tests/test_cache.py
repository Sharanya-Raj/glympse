import json

from src.cache import JsonCache
from src.models import SOURCE_DB_DIRECT, SOURCE_DB_MISS, SOURCE_LLM, SOURCE_MANUAL, SOURCE_UNRESOLVED, NutritionResult
from src.nutrition_lookup import ManualOverrideProvider, NutritionResolver

from .conftest import FakeProvider


class FakeEstimator:
    model = "fake-llm"

    def __init__(self):
        self.requested: list[list[str]] = []

    def estimate(self, names, on_batch=None):
        self.requested.append(list(names))
        results = {
            n: NutritionResult(
                normalized_name=n, calories_per_100g=200, carbs_per_100g=10, protein_per_100g=15,
                fat_per_100g=11, source=SOURCE_LLM, confidence=0.55, reasoning_summary="typical dish",
            )
            for n in names
        }
        if on_batch is not None:
            on_batch(results)
        return results


def test_json_cache_roundtrip_and_atomic_write(tmp_path):
    path = tmp_path / "c.json"
    cache = JsonCache(path, autosave_every=2)
    cache.set("a", {"x": 1})
    assert not path.exists()  # buffered until autosave_every writes
    cache.set("b", {"x": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"x": 1}, "b": {"x": 2}}
    assert not path.with_name("c.json.tmp").exists()
    assert JsonCache(path).get("b") == {"x": 2}


def test_corrupt_cache_is_moved_aside_not_deleted(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    cache = JsonCache(path)
    assert len(cache) == 0
    backups = list(tmp_path.glob("c.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "{not json"


def _resolver(settings, provider, estimator=None, refresh=False):
    cache = JsonCache(settings.nutrition_cache_path)
    overrides = ManualOverrideProvider(settings.overrides_path)
    return NutritionResolver(cache, settings, [provider], overrides=overrides, estimator=estimator, refresh=refresh)


def test_resolved_foods_are_cached_and_not_looked_up_again(settings):
    provider = FakeProvider({"orange": (47, 11.8, 0.9, 0.1)})
    first = _resolver(settings, provider).resolve(["orange"])
    assert first["orange"].source == SOURCE_DB_DIRECT
    assert provider.queries == ["orange"]

    provider2 = FakeProvider({})
    second = _resolver(settings, provider2).resolve(["orange"])
    assert second["orange"].calories_per_100g == 47
    assert provider2.queries == []


def test_db_miss_goes_to_llm_and_keeps_rejected_candidate(settings):
    estimator = FakeEstimator()
    results = _resolver(settings, FakeProvider({}), estimator).resolve(["braised duck wing"])
    r = results["braised duck wing"]
    assert r.source == SOURCE_LLM
    assert r.db_candidate_name == "Something else" and r.db_candidate_score == 0.2
    assert estimator.requested == [["braised duck wing"]]


def test_skip_llm_fallback_caches_db_miss_for_later_llm_run(settings):
    provider = FakeProvider({})
    results = _resolver(settings, provider).resolve(["mystery dish"], use_llm_fallback=False)
    assert results["mystery dish"].source == SOURCE_UNRESOLVED
    assert JsonCache(settings.nutrition_cache_path).get("mystery dish")["source"] == SOURCE_DB_MISS

    provider2 = FakeProvider({})
    estimator = FakeEstimator()
    results = _resolver(settings, provider2, estimator).resolve(["mystery dish"])
    assert results["mystery dish"].source == SOURCE_LLM
    assert provider2.queries == []  # the database is not asked again


def test_manual_override_wins_and_is_never_written_to_cache(settings):
    settings.overrides_path.write_text(
        "normalized_food_name,calories_per_100g,carbs_per_100g,protein_per_100g,fat_per_100g,notes\n"
        "Orange,40,10,1,0,checked by dietitian\n",
        encoding="utf-8",
    )
    cache = JsonCache(settings.nutrition_cache_path)
    cache.set("orange", {"normalized_name": "orange", "calories_per_100g": 99, "carbs_per_100g": 1,
                         "protein_per_100g": 1, "fat_per_100g": 1, "source": SOURCE_LLM, "confidence": 0.5})
    provider = FakeProvider({"orange": (47, 11.8, 0.9, 0.1)})
    results = _resolver(settings, provider, refresh=True).resolve(["orange"])
    r = results["orange"]
    assert (r.source, r.confidence, r.calories_per_100g) == (SOURCE_MANUAL, 1.0, 40)
    assert r.notes == "checked by dietitian"
    assert provider.queries == []
    assert JsonCache(settings.nutrition_cache_path).get("orange")["source"] == SOURCE_LLM


def test_provider_error_is_not_cached(settings):
    class Broken:
        name = "broken"

        def search(self, food_name):
            raise ConnectionError("down")

    results = _resolver(settings, Broken(), FakeEstimator()).resolve(["rice"])
    assert results["rice"].source == SOURCE_UNRESOLVED
    assert "database_lookup_error" in results["rice"].flags
    assert JsonCache(settings.nutrition_cache_path).get("rice") is None
