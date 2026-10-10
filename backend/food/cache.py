"""Offline-first food data: the pre-built label->fdcId map and nutrient cache committed under
backend/food/data/, plus a small bounded in-memory cache for live USDA lookups at runtime."""
import json
import threading
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
FOOD_MAP_FILE = "food_map.json"
NUTRIENT_CACHE_FILE = "nutrient_cache.json"


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return default


class FoodCache:
    """food_map: label -> {fdcId, description, dataType}. nutrients: str(fdcId) -> nutrition record."""

    def __init__(self, data_dir=None, max_live=256):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.food_map = load_json(self.data_dir / FOOD_MAP_FILE, {})
        self.nutrients = load_json(self.data_dir / NUTRIENT_CACHE_FILE, {})
        self._live = {}
        self._max_live = max_live
        self._lock = threading.Lock()

    def map_label(self, label):
        """The mapped USDA food for a classifier label, if one was pre-mapped."""
        return self.food_map.get(label) or self.food_map.get(label.lower())

    def nutrition_for(self, fdc_id):
        """Cached nutrition record for an fdcId (disk cache first, then anything learned live)."""
        key = str(fdc_id)
        return self.nutrients.get(key) or self._live.get(("nutrition", key))

    def remember_nutrition(self, record):
        self._store(("nutrition", str(record["fdcId"])), record)

    def recall_search(self, query):
        return self._live.get(("search", query.strip().lower()))

    def remember_search(self, query, records):
        self._store(("search", query.strip().lower()), records)

    def _store(self, key, value):
        with self._lock:
            if key not in self._live and len(self._live) >= self._max_live:
                self._live.pop(next(iter(self._live)))
            self._live[key] = value
