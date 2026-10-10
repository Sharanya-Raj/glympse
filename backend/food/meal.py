"""Voice-driven meal mapping: speech (ElevenLabs) + optional photo (Gemini) -> USDA foods.

Used by the dictation API and `scripts/build_food_map.py --record / --audio`.
sounddevice and google-genai are imported lazily so the label-mapping flow runs without them.

Speech is the source of truth: the photo only clarifies foods the speaker mentioned. Quantities
the speaker did not state stay null, and serving nutrition is computed only from a stated weight
or a compatible USDA household portion; per-100 g data is always kept.
"""
import io
import json
import mimetypes
import os
import re
import time
import uuid
import wave
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from .identify import ALLOWED_FORMATS
from .nutrition import scale_nutrients
from .usda import UsdaError

SCHEMA_VERSION = 1
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_MODEL = "scribe_v2"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_API_AI")
MAX_RECORD_SECONDS = 120
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_SIDE = 1536
CANDIDATE_LIMIT = 5
HTTP_TIMEOUT = 60.0
RETRIES = 2  # extra attempts after the first, for timeouts / 429 / 5xx
RETRY_DELAY = 1.0
RETRY_STATUS = (408, 429, 500, 502, 503, 504)

WEIGHT_UNITS = {"g": 1.0, "gram": 1.0, "gm": 1.0, "kg": 1000.0, "kilogram": 1000.0,
                "oz": 28.3495, "ounce": 28.3495, "lb": 453.592, "pound": 453.592}
UNIT_ALIASES = {"tbsp": "tablespoon", "tbs": "tablespoon", "tsp": "teaspoon",
                "pc": "piece", "pcs": "piece", "grams": "gram", "lbs": "lb"}


class MealError(Exception):
    """A meal-mode step failed in a way the user can act on; the message says what to do."""


# --- Configuration ---------------------------------------------------------------------

def elevenlabs_key():
    key = os.environ.get("ELEVEN_LABS_API_KEY", "").strip()
    if not key:
        raise MealError("ElevenLabs API key not configured: set ELEVEN_LABS_API_KEY in backend/.env")
    return key


def gemini_key():
    for name in GEMINI_KEY_VARS:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    raise MealError("Gemini API key not configured: set GEMINI_API_KEY in backend/.env")


def gemini_model():
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


# --- Audio -----------------------------------------------------------------------------

def _sounddevice():
    try:
        import sounddevice
    except ImportError as exc:
        raise MealError("Recording needs sounddevice: python -m pip install sounddevice") from exc
    except OSError as exc:  # sounddevice imports, but the PortAudio library is missing
        raise MealError(f"Could not load the PortAudio audio library: {exc}") from exc
    return sounddevice


def encode_wav(pcm, sample_rate):
    """Mono 16-bit PCM bytes -> WAV file bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def record_wav(stop_event, max_seconds=MAX_RECORD_SECONDS, sd=None):
    """Record the default microphone until `stop_event` is set or `max_seconds` pass.

    Returns (wav_bytes, seconds). The stream is always closed; KeyboardInterrupt propagates.
    """
    sd = sd or _sounddevice()
    try:
        rate = int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception as exc:
        raise MealError(f"No usable microphone found ({exc}). Check that one is connected and "
                        "that microphone access is allowed in Windows privacy settings.") from exc
    chunks = []

    def on_audio(indata, frames, time_info, status):
        chunks.append(bytes(indata))

    try:
        stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", callback=on_audio)
    except Exception as exc:
        raise MealError(f"Could not open the microphone: {exc}") from exc
    try:
        try:
            stream.start()
        except Exception as exc:
            raise MealError(f"Could not start recording from the microphone: {exc}") from exc
        deadline = time.monotonic() + max_seconds
        # Short waits keep Ctrl+C responsive on Windows, where a long Event.wait is not interruptible.
        while not stop_event.wait(0.1) and time.monotonic() < deadline:
            pass
    finally:
        stream.close()
    pcm = b"".join(chunks)
    if not pcm:
        raise MealError("No audio was captured from the microphone.")
    return encode_wav(pcm, rate), len(pcm) / 2 / rate


def read_audio_file(path):
    """(bytes, filename, content_type) for an existing recording."""
    path = Path(path)
    if not path.is_file():
        raise MealError(f"Audio file not found: {path}")
    data = path.read_bytes()
    if not data:
        raise MealError(f"Audio file is empty: {path}")
    if len(data) > MAX_AUDIO_BYTES:
        raise MealError(f"Audio file is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB: {path}")
    return data, path.name, mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _post_with_retries(client, url, service, **kwargs):
    error = None
    for attempt in range(RETRIES + 1):
        try:
            response = client.post(url, **kwargs)
        except httpx.TransportError as exc:  # includes timeouts
            error = MealError(f"Could not reach {service}: {exc}")
        else:
            if response.status_code not in RETRY_STATUS:
                return response
            error = MealError(f"{service} is unavailable (HTTP {response.status_code}); try again shortly.")
        if attempt < RETRIES:
            time.sleep(RETRY_DELAY * 2 ** attempt)
    raise error


def transcribe(audio, filename, content_type, api_key, client=None):
    """Speech-to-text with ElevenLabs Scribe; language auto-detected, no diarization or audio tags."""
    own = client is None
    client = client or httpx.Client(timeout=HTTP_TIMEOUT)
    try:
        response = _post_with_retries(
            client, ELEVENLABS_URL, "ElevenLabs",
            headers={"xi-api-key": api_key},
            data={"model_id": STT_MODEL, "diarize": "false", "tag_audio_events": "false"},
            files={"file": (filename, audio, content_type)})
    finally:
        if own:
            client.close()
    if response.status_code in (401, 403):
        raise MealError(f"ElevenLabs rejected the API key (HTTP {response.status_code}); "
                        "check ELEVEN_LABS_API_KEY in backend/.env")
    if response.status_code != 200:
        snippet = response.text[:200].replace("\n", " ").strip()
        raise MealError(f"ElevenLabs transcription failed (HTTP {response.status_code}): {snippet}")
    try:
        text = (response.json().get("text") or "").strip()
    except (ValueError, AttributeError) as exc:
        raise MealError("ElevenLabs returned an unreadable response.") from exc
    if not text:
        raise MealError("The transcript is empty; speak closer to the microphone or check the audio file.")
    return text


# --- Photo -----------------------------------------------------------------------------

def prepare_image(path):
    """Validated, EXIF-rotated JPEG bytes no larger than MAX_IMAGE_SIDE pixels on a side."""
    path = Path(path)
    if not path.is_file():
        raise MealError(f"Image file not found: {path}")
    data = path.read_bytes()
    if not data:
        raise MealError(f"Image file is empty: {path}")
    if len(data) > MAX_IMAGE_BYTES:
        raise MealError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB: {path}")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise MealError(f"{path} could not be read as an image; use a JPG, PNG or WebP photo.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise MealError(f"{path} is {image.format or 'an unknown format'}; use a JPG, PNG or WebP photo.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()


# --- Gemini ----------------------------------------------------------------------------

class MealFood(BaseModel):
    name: str = Field(description="The food as the speaker named it, e.g. 'scrambled eggs'.")
    preparation: Optional[str] = Field(
        description="Cooking method or style stated by the speaker or clearly visible in the photo; null otherwise.")
    quantity: Optional[float] = Field(description="Amount the speaker stated, as a number; null if not stated.")
    unit: Optional[str] = Field(description="Unit of the stated quantity (g, cup, slice...); null if none was stated.")
    search_terms: str = Field(description="Short generic USDA FoodData Central search query, e.g. 'egg scrambled'.")
    source: Literal["speech", "speech+photo"] = Field(
        description="'speech+photo' when the photo shows this food, otherwise 'speech'.")
    warnings: list[str] = Field(description="Notes about this item, e.g. the photo disagrees with the speech.")


class MealInterpretation(BaseModel):
    items: list[MealFood] = Field(max_length=20)
    warnings: list[str] = Field(description="Meal-level notes, e.g. the photo shows a different meal.")


class CandidateChoice(BaseModel):
    item_index: int
    fdc_id: Optional[int] = Field(description="Chosen candidate fdc_id, or null if none is a reasonable match.")


class CandidateChoices(BaseModel):
    choices: list[CandidateChoice]


class PortionChoice(BaseModel):
    item_index: int
    portion_index: Optional[int] = Field(description="Index from this food's USDA portions, or null if incompatible.")


class PortionChoices(BaseModel):
    choices: list[PortionChoice]


INTERPRET_PROMPT = """You turn a spoken description of a meal into a structured food list for a USDA FoodData Central lookup.

Rules:
- The transcript is the source of truth. Include only foods the speaker says they ate or are about to eat.
- Apply corrections and exclusions: "two eggs, actually three" means 3 eggs; "I didn't eat the fries" means no fries item.
- {photo_rule}
- Keep a mixed dish (lasagna, burrito, fried rice) as one item unless the speaker explicitly lists its components as separate foods.
- quantity/unit: preserve the actual spoken amount ("three eggs" -> 3 and "egg"; "two slices of pizza" -> 2 and "slice"; "a cup of rice" -> 1 and "cup"; "200 grams of chicken" -> 200 and "g"). A counted food is a unit, even without a separate measurement word. Never replace a stated count with 1. Use null when not stated. Never estimate weights or serving sizes.
- Return one row per distinct food with its full quantity, not one row per physical piece. Preserve every distinct food in the transcript.
- search_terms: a short generic food name for USDA search, without brands or quantities, e.g. "egg scrambled", "rice white cooked".
- Never invent ingredients, weights, USDA identifiers or nutrient values.
- If the transcript mentions no food, return an empty items list and explain in warnings.

Transcript:
\"\"\"{transcript}\"\"\"
"""
PHOTO_RULE = ("A photo of the meal is attached. Use it only to clarify foods the speaker mentioned (type, "
              "preparation). Never add foods that appear only in the photo. If the photo disagrees with the "
              "speech, follow the speech and add a warning.")
NO_PHOTO_RULE = "No photo is attached, so every item's source is \"speech\"."

CHOOSE_PROMPT = """For each meal item below, choose the USDA FoodData Central candidate that is the same food with the closest preparation, or null if no candidate is a reasonable match.

Rules:
- Use only an fdc_id from that item's own candidate list.
- Food identity matters most, then preparation (a "fried egg" should not match "egg, raw").
- Return one choice per item_index.

Items:
{items}
"""


def parse_model_output(schema, text):
    """Validate a model's JSON text against `schema`, or raise MealError."""
    try:
        return schema.model_validate_json(text or "")
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first.get("loc", ())) or "response"
        raise MealError(f"Gemini returned output in an unexpected format ({where}: {first['msg']}).") from exc


class Gemini:
    """Structured-output Gemini calls with bounded timeouts and transient retries."""

    def __init__(self, api_key, model=None, timeout=HTTP_TIMEOUT):
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as exc:
            raise MealError("Meal interpretation needs google-genai: python -m pip install google-genai") from exc
        self.model = model or gemini_model()
        self._types, self._errors = types, errors
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000),
            retry_options=types.HttpRetryOptions(attempts=RETRIES + 1, http_status_codes=list(RETRY_STATUS))))

    def close(self):
        self._client.close()

    def generate_json(self, prompt, schema, image=None):
        types = self._types
        contents = ([types.Part.from_bytes(data=image, mime_type="image/jpeg")] if image else []) + [prompt]
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             response_json_schema=schema.model_json_schema())
        try:
            response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except self._errors.APIError as exc:
            message = exc.message or str(exc)
            if exc.code in (401, 403) or "API key" in message:
                raise MealError(f"Gemini rejected the API key (HTTP {exc.code}); check GEMINI_API_KEY") from exc
            if exc.code == 404:
                raise MealError(f"Gemini model {self.model!r} was not found; set GEMINI_MODEL") from exc
            raise MealError(f"Gemini request failed (HTTP {exc.code}): {message}") from exc
        except httpx.HTTPError as exc:
            raise MealError(f"Could not reach Gemini: {exc}") from exc
        return parse_model_output(schema, response.text)


def interpret_meal(transcript, gemini, image=None):
    """Transcript (+ optional photo) -> MealInterpretation with at least one food."""
    prompt = INTERPRET_PROMPT.format(photo_rule=PHOTO_RULE if image else NO_PHOTO_RULE, transcript=transcript)
    result = gemini.generate_json(prompt, MealInterpretation, image=image)
    if not image:
        for food in result.items:
            food.source = "speech"
    if not result.items:
        detail = f" ({'; '.join(result.warnings)})" if result.warnings else ""
        raise MealError(f"No foods were recognized in the transcript{detail}.")
    return result


# --- USDA matching ---------------------------------------------------------------------

def normalize_label(text):
    """'French fries' / 'french-fries' -> 'french_fries', the classifier label style in food_map.json."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _usda_match(entry):
    return {"fdcId": entry["fdcId"], "description": entry.get("description"), "dataType": entry.get("dataType")}


def _new_item(food):
    return {"name": food.name, "quantity": food.quantity, "unit": food.unit, "preparation": food.preparation,
            "source": food.source, "search_terms": food.search_terms, "match_status": "unmatched",
            "match": None, "warnings": list(food.warnings)}


def choose_candidates(foods, pending, gemini):
    """{item_index: fdcId or None}, with every chosen id checked against that item's candidates."""
    listing = [{"item_index": index, "name": foods[index].name, "preparation": foods[index].preparation,
                "candidates": [{"fdc_id": c["fdcId"], "description": c.get("description"),
                                "dataType": c.get("dataType")} for c in found]}
               for index, found in pending.items()]
    result = gemini.generate_json(CHOOSE_PROMPT.format(items=json.dumps(listing, indent=2)), CandidateChoices)
    chosen, rejected = {}, set()
    for choice in result.choices:
        if choice.item_index not in pending or choice.item_index in chosen:
            continue
        allowed = {c["fdcId"] for c in pending[choice.item_index]}
        if choice.fdc_id is not None and choice.fdc_id not in allowed:
            rejected.add(choice.item_index)
            chosen[choice.item_index] = None
        else:
            chosen[choice.item_index] = choice.fdc_id
    return chosen, rejected


def match_items(foods, food_map, search, gemini):
    """Ordered meal items, each with match_status mapped | matched | unmatched | error.

    `search(term)` returns USDA candidates ({fdcId, description, dataType, ...}) or raises UsdaError.
    """
    items, pending = [], {}
    for index, food in enumerate(foods):
        item = _new_item(food)
        entry = food_map.get(normalize_label(food.search_terms)) or food_map.get(normalize_label(food.name))
        if entry:
            item["match_status"], item["match"] = "mapped", _usda_match(entry)
        else:
            try:
                found = [c for c in search(food.search_terms) if c.get("fdcId") is not None][:CANDIDATE_LIMIT]
            except UsdaError as exc:
                item["match_status"] = "error"
                item["warnings"].append(f"USDA search failed: {exc}")
            else:
                if found:
                    pending[index] = found
                else:
                    item["warnings"].append(f"No USDA results for '{food.search_terms}'.")
        items.append(item)
    if not pending:
        return items
    try:
        chosen, rejected = choose_candidates(foods, pending, gemini)
    except MealError as exc:
        for index in pending:
            items[index]["warnings"].append(f"Could not choose a USDA match: {exc}")
        return items
    for index, found in pending.items():
        item, fdc_id = items[index], chosen.get(index)
        if index in rejected:
            item["warnings"].append("Gemini chose a USDA food outside the search results; left unmatched.")
        elif fdc_id is None:
            item["warnings"].append(f"None of the USDA results for '{item['search_terms']}' matched this food.")
        else:
            item["match_status"] = "matched"
            item["match"] = _usda_match(next(c for c in found if c["fdcId"] == fdc_id))
    return items


# --- Nutrition -------------------------------------------------------------------------

def _unit_key(text):
    word = re.sub(r"[^a-z]", "", (text or "").lower())
    word = UNIT_ALIASES.get(word, word)
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _portion_amount(label):
    """Reference amount and primary unit; descriptions like '(8 slices)' are not units."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?(?:\s+\d+/\d+)?|\d+/\d+)\s+(.+)$", label or "")
    if not match:
        return None, set()
    amount = sum(Fraction(part) for part in match.group(1).split())
    measure = re.split(r"[,()]|\s+of\s+", match.group(2), maxsplit=1)[0]
    return amount, {_unit_key(word) for word in measure.split()}


def serving_grams(quantity, unit, portions):
    """(grams, basis) from a stated weight or a USDA portion with the same unit, else (None, None)."""
    if quantity is None or quantity <= 0 or not unit:
        return None, None
    key = _unit_key(unit)
    if key in WEIGHT_UNITS:
        return round(quantity * WEIGHT_UNITS[key], 1), "stated weight"
    # Prefer an unqualified reference over special cases such as 'crust not eaten'.
    for portion in sorted(portions, key=lambda p: len(p.get("label", ""))):
        amount, words = _portion_amount(portion.get("label"))
        if amount and key in words:
            return round(quantity * portion["grams"] / float(amount), 1), f"USDA portion '{portion['label']}'"
    return None, None


def choose_portions(items, records, gemini):
    """Resolve counted foods (e.g. eggs -> USDA '1 large') without inventing weights."""
    pending = {}
    for index, item in enumerate(items):
        record = records.get(str((item.get("match") or {}).get("fdcId")), {})
        portions = record.get("portions") or []
        if not item["quantity"] or item["quantity"] <= 0 or not portions:
            continue
        grams, basis = serving_grams(item["quantity"], item["unit"], portions)
        same_unit = [p for p in portions if _unit_key(item["unit"]) in _portion_amount(p["label"])[1]]
        unit_weights = {round(p["grams"] / float(_portion_amount(p["label"])[0]), 4)
                        for p in same_unit if _portion_amount(p["label"])[0]}
        if basis == "stated weight" or (grams is not None and len(unit_weights) == 1):
            continue
        choices = []
        for p_index, portion in enumerate(portions):
            amount, units = _portion_amount(portion["label"])
            if same_unit and _unit_key(item["unit"]) not in units:
                continue
            if amount and amount > 0:
                choices.append({"portion_index": p_index, "label": portion["label"]})
        if choices:
            pending[index] = {"item_index": index, "food": item["name"], "quantity": item["quantity"],
                              "unit": item["unit"], "portions": choices}
    if not pending:
        return {}
    prompt = """Match each spoken food unit to ONE compatible USDA reference portion from its own list.
Return only a portion_index, never a weight or changed quantity. The application multiplies the
spoken quantity by the USDA weight divided by the reference portion's numeric amount.
For a counted food with no unit, use its food name as the unit (3 eggs means 3 individual eggs).
Examples: eggs can match '1 large'; bananas can match '1 medium'; burgers can match '1 sandwich'.
Whole food counts must not match cups, tablespoons, grams, slices, chopped pieces, or servings
of unspecified size. A bowl must not match a cup. Never treat a slice as a whole pizza or loaf.
Respect any spoken size/preparation. If size is unspecified, prefer medium, regular, or a
standard individual item (large for eggs). If no compatible reference exists, choose null.
Unless explicitly stated, choose the complete food, not 'crust not eaten', peeled-off parts,
or a partially eaten portion. Prefer a plain standard slice when slice size is unspecified.
Use only an index in that item's list; return one choice per item_index.
Items:
""" + json.dumps(list(pending.values()))
    try:
        response = gemini.generate_json(prompt, PortionChoices)
    except MealError as exc:
        for index in pending:
            items[index]["warnings"].append(f"Could not resolve the spoken portion: {exc}")
        return {}
    selected = {}
    for choice in response.choices:
        if choice.item_index in pending and choice.portion_index in {
                p["portion_index"] for p in pending[choice.item_index]["portions"]}:
            selected[choice.item_index] = choice.portion_index
    return selected


def attach_nutrition(items, records, gemini=None):
    """Embed per-100 g nutrition (and a serving when the amount is known) into matched items.

    `records` maps str(fdcId) -> nutrition record from nutrition.build_nutrition.
    """
    chosen = choose_portions(items, records, gemini) if gemini else {}
    for index, item in enumerate(items):
        item["nutrition"] = None
        if not item["match"]:
            continue
        record = records.get(str(item["match"]["fdcId"]))
        if not record:
            item["warnings"].append("Nutrition is unavailable for this USDA food.")
            continue
        portions = record.get("portions") or []
        grams, basis = serving_grams(item["quantity"], item["unit"], portions)
        portion_index = chosen.get(index)
        if portion_index is None and basis and basis.startswith("USDA portion"):
            portion_index = next(i for i, p in enumerate(portions) if basis == f"USDA portion '{p['label']}'")
        portion_count = None
        if portion_index is not None:
            portion = portions[portion_index]
            reference_amount, _ = _portion_amount(portion["label"])
            portion_count = item["quantity"] / float(reference_amount)
            grams = round(portion_count * portion["grams"], 1)
            basis = f"USDA portion '{portion['label']}'"
        serving = None
        if grams is not None:
            serving = {"grams": grams, "basis": basis, "nutrients": scale_nutrients(record["per_100g"], grams),
                       "portion_index": portion_index, "portion_count": portion_count}
        item["nutrition"] = {"per_100g": record["per_100g"], "portions": record.get("portions") or [],
                             "default_portion": record.get("default_portion"),
                             "missing": record.get("missing", []),
                             "kcal_estimated": record.get("kcal_estimated", False), "serving": serving}
    return items


# --- Output ----------------------------------------------------------------------------

def build_meal(transcript, image_used, interpretation, items, model, nutrition_included):
    return {"schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "transcript": transcript, "image_used": bool(image_used), "interpretation_model": model,
            "nutrition_included": bool(nutrition_included),
            "warnings": list(interpretation.warnings), "items": items}


def default_meal_path(repo_root):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(repo_root) / "outputs" / "meals" / f"{stamp}-{uuid.uuid4().hex[:8]}.json"


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
