"""FastAPI router for the photo/typed food feature. Mounted by backend.app.create_app.

Endpoints (no /api prefix, matching the rest of this app):
  POST /food/identify          multipart "image"  -> predictions + needs_confirmation
  POST /food/dictate           multipart "audio"  -> transcript + foods + USDA nutrition
  GET  /food/nutrients?label=  classifier label   -> nutrition record (cache-first)
  GET  /food/nutrients/{fdcId} USDA fdcId          -> nutrition record
  GET  /food/search?q=         free text           -> up to 10 matches (powers "none of these")

The USDA key never leaves the server; these endpoints proxy FDC. HTTP status mapping is chosen
so the existing frontend api.js preserves the detail message (400/404/422/502) and so the
photo model-missing case (503) can be handled specifically in the UI.
"""
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from . import identify as idf
from . import usda
from .cache import FoodCache
from .nutrition import build_nutrition

MAX_DICTATION_BYTES = 10 * 1024 * 1024
AUDIO_TYPES = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a",
               "audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3"}


def build_food_router(user_dependency, cache=None, threshold=None):
    cache = cache if cache is not None else FoodCache()
    threshold = idf.DEFAULT_THRESHOLD if threshold is None else threshold
    router = APIRouter(prefix="/food", tags=["food"])

    def nutrition_from_detail(fdc_id, description="", data_type=""):
        """Cache-first: return the cached record if present (no network), else fetch detail once."""
        cached = cache.nutrition_for(fdc_id)
        if cached is not None:
            return cached
        try:
            detail = usda.food_detail(fdc_id)
        except usda.UsdaError as exc:
            raise HTTPException(502, f"USDA lookup failed: {exc}")
        record = build_nutrition(detail.get("fdcId", fdc_id), detail.get("description") or description,
                                 detail.get("dataType") or data_type, detail.get("foodNutrients"), detail)
        cache.remember_nutrition(record)
        return record

    def resolve_label(label):
        entry = cache.map_label(label)
        if entry:
            return nutrition_from_detail(entry["fdcId"], entry.get("description", ""), entry.get("dataType", ""))
        display = label.replace("_", " ").strip()
        try:
            foods = usda.search_foods(display, page_size=1)
        except usda.UsdaError as exc:
            raise HTTPException(502, f"USDA search failed: {exc}")
        if not foods:
            raise HTTPException(404, f"No USDA match for '{display}'.")
        top = foods[0]
        return nutrition_from_detail(top.get("fdcId"), top.get("description", ""), top.get("dataType", ""))

    @router.post("/dictate")
    def dictate(audio: UploadFile = File(...), user_id=Depends(user_dependency)):
        # A synchronous route keeps provider calls/retries off the ASGI event loop.
        from . import meal

        mime = (audio.content_type or "").split(";")[0].strip().lower()
        if mime not in AUDIO_TYPES:
            raise HTTPException(415, "Use WebM, Ogg, MP4, WAV, or MP3 audio.")
        content = audio.file.read(MAX_DICTATION_BYTES + 1)
        if len(content) > MAX_DICTATION_BYTES:
            raise HTTPException(413, "Dictation must be at most 10 MB.")
        if not content:
            raise HTTPException(422, "No audio captured. Record your meal again.")
        try:
            stt_key = meal.elevenlabs_key()
            gemini_key = meal.gemini_key()
            usda.api_key()
            gemini = meal.Gemini(gemini_key)
        except (meal.MealError, usda.UsdaError) as exc:
            raise HTTPException(503, str(exc)) from exc
        try:
            transcript = meal.transcribe(content, "meal." + AUDIO_TYPES[mime], mime, stt_key)
            interpretation = meal.interpret_meal(transcript, gemini)
            items = meal.match_items(interpretation.items, cache.food_map,
                                    lambda term: usda.search_foods(term, page_size=meal.CANDIDATE_LIMIT), gemini)
            records = {}
            for item in items:
                match = item["match"]
                if match and str(match["fdcId"]) not in records:
                    try:
                        records[str(match["fdcId"])] = nutrition_from_detail(
                            match["fdcId"], match.get("description", ""), match.get("dataType", ""))
                    except HTTPException as exc:
                        item["warnings"].append(str(exc.detail))
            meal.attach_nutrition(items, records, gemini=gemini)
            return meal.build_meal(transcript, False, interpretation, items, gemini.model, True)
        except meal.MealError as exc:
            raise HTTPException(502, str(exc)) from exc
        finally:
            gemini.close()

    @router.post("/identify")
    async def food_identify(image: UploadFile = File(...), user_id=Depends(user_dependency)):
        content = await image.read(idf.MAX_UPLOAD_BYTES + 1)
        if len(content) > idf.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image must be at most 10 MB.")
        try:
            return idf.identify(content, image.content_type, threshold)
        except idf.ImageValidationError as exc:
            raise HTTPException(400, str(exc))
        except idf.ModelUnavailable as exc:
            raise HTTPException(503, str(exc))

    @router.get("/nutrients")
    def nutrients_by_label(label: str = Query(..., min_length=1, max_length=100),
                           user_id=Depends(user_dependency)):
        return resolve_label(label)

    @router.get("/nutrients/{fdc_id}")
    def nutrients_by_id(fdc_id: int, user_id=Depends(user_dependency)):
        return nutrition_from_detail(fdc_id)

    @router.get("/search")
    def food_search(q: str = Query(..., min_length=1, max_length=100), user_id=Depends(user_dependency)):
        cached = cache.recall_search(q)
        if cached is not None:
            return {"query": q, "results": cached}
        try:
            foods = usda.search_foods(q, page_size=10)
        except usda.UsdaError as exc:
            raise HTTPException(502, f"USDA search failed: {exc}")
        # Search records carry nutrients but not portions; portions arrive when one is selected
        # via /food/nutrients/{fdcId}. This keeps search to a single USDA call.
        results = [build_nutrition(f.get("fdcId"), f.get("description"), f.get("dataType"),
                                   f.get("foodNutrients"), None) for f in foods[:10]]
        cache.remember_search(q, results)
        return {"query": q, "results": results}

    return router
