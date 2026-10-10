"""Thin USDA FoodData Central client. POST for search (GET is rejected by USDA's nginx on
some queries), GET for detail. The API key comes from USDA_API_KEY (or FDC_API_KEY)."""
import os

import httpx

BASE = os.environ.get("USDA_FDC_BASE", "https://api.nal.usda.gov/fdc/v1").rstrip("/")
# Photo-identified foods are generic dishes, so never Branded; FNDDS + SR Legacy only.
SEARCH_DATATYPES = ["Survey (FNDDS)", "SR Legacy"]
TIMEOUT = float(os.environ.get("USDA_FDC_TIMEOUT", "20"))


class UsdaError(Exception):
    """A USDA call failed (no key, network error, or a non-200 response)."""
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def api_key():
    key = (os.environ.get("USDA_API_KEY") or os.environ.get("FDC_API_KEY") or "").strip()
    if not key:
        raise UsdaError("USDA API key not configured (set USDA_API_KEY in backend/.env)")
    return key


def _call(method, path, client, **kwargs):
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT)
    try:
        response = client.request(method, f"{BASE}{path}",
                                  params={"api_key": api_key(), **kwargs.pop("params", {})}, **kwargs)
    except httpx.HTTPError as exc:
        raise UsdaError(f"Could not reach USDA FoodData Central: {exc}") from exc
    finally:
        if own:
            client.close()
    if response.status_code != 200:
        snippet = response.text[:200].replace("\n", " ").strip()
        raise UsdaError(f"USDA returned HTTP {response.status_code}: {snippet}", response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise UsdaError("USDA returned a non-JSON response") from exc


def search_foods(query, page_size=10, data_types=None, client=None):
    """Up to `page_size` foods matching `query`, newest-schema shape. POST avoids the nginx 400."""
    data = _call("POST", "/foods/search", client,
                 json={"query": query, "dataType": data_types or SEARCH_DATATYPES,
                       "pageSize": page_size, "requireAllWords": False})
    return data.get("foods", [])


def food_detail(fdc_id, client=None):
    """Full detail record for one fdcId, including foodNutrients and foodPortions."""
    return _call("GET", f"/food/{fdc_id}", client)
