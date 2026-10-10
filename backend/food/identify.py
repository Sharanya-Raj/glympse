"""Local food-photo identification with the nateraw/food Vision Transformer (Food-101).

No generative/LLM calls: this is a fixed 101-class image classifier loaded through the
transformers image-classification pipeline. The model (~500 MB) downloads once on first use
and is kept in memory. torch + transformers are optional at import time so the rest of the
app runs without them; a clean ModelUnavailable is raised if identification is attempted
without them installed.
"""
import io
import logging
import os
import threading

from PIL import Image, UnidentifiedImageError

log = logging.getLogger("glyimpse.food")

MODEL_NAME = os.environ.get("FOOD_MODEL_NAME", "nateraw/food")
DEFAULT_THRESHOLD = float(os.environ.get("FOOD_CONFIDENCE_THRESHOLD", "0.6"))
TOP_K = 5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
# Pillow's format names for the types we accept.
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"}  # MPO: some phone JPEGs report this

_pipeline = None
_lock = threading.Lock()


class ImageValidationError(ValueError):
    """The uploaded file is not a usable jpg/png/webp image within the size limit."""


class ModelUnavailable(RuntimeError):
    """The classifier could not be loaded (torch/transformers missing or download failed)."""


def display_name(label):
    """Human label from a snake_case class, e.g. 'french_fries' -> 'French Fries'."""
    return " ".join(word.capitalize() for word in label.replace("-", "_").split("_") if word)


def validate_image(content, content_type=None):
    """Return an RGB PIL image, or raise ImageValidationError with a clear reason."""
    if not content:
        raise ImageValidationError("The uploaded image is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Image must be at most 10 MB.")
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError("Only JPG, PNG or WebP images are supported.")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("That file could not be read as an image.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise ImageValidationError(f"Unsupported image format ({image.format or 'unknown'}); use JPG, PNG or WebP.")
    return image.convert("RGB")


def get_classifier():
    """Lazily build (and cache) the transformers image-classification pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ModelUnavailable(
                "Food photo identification needs torch + transformers. Install them with "
                "pip install -r backend/food/requirements.txt") from exc
        log.info("Loading food classifier %s (first run downloads ~500 MB) ...", MODEL_NAME)
        try:
            _pipeline = pipeline("image-classification", model=MODEL_NAME, top_k=None)
        except Exception as exc:  # network/download/load failure
            raise ModelUnavailable(f"Could not load the food classifier: {exc}") from exc
        log.info("Food classifier ready: %d classes", len(_pipeline.model.config.id2label))
        return _pipeline


def _predictions(raw):
    """Top-K {label, display_name, confidence} from the pipeline's scored list."""
    ranked = sorted(raw, key=lambda item: item["score"], reverse=True)[:TOP_K]
    return [{"label": item["label"], "display_name": display_name(item["label"]),
             "confidence": round(float(item["score"]), 4)} for item in ranked]


def identify(content, content_type=None, threshold=DEFAULT_THRESHOLD, classifier=None):
    """Validate the upload, classify it, and return top-5 predictions + needs_confirmation.

    `classifier` is injectable for tests; production passes None and the lazy pipeline is used.
    """
    image = validate_image(content, content_type)
    classifier = classifier or get_classifier()
    raw = classifier(image)
    predictions = _predictions(raw)
    if not predictions:
        raise ModelUnavailable("The classifier returned no predictions.")
    needs_confirmation = predictions[0]["confidence"] < threshold
    return {"predictions": predictions, "needs_confirmation": needs_confirmation}
