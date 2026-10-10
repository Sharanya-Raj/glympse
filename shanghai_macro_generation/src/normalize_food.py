"""Deterministic, conservative food-name normalization (no synonym merging)."""

from __future__ import annotations

import re
import unicodedata

_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_DASH_RE = re.compile(f"[{re.escape(_DASHES)}]")
_AMOUNT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:kg|g|grams?|mg|ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|fl\.?\s*oz|oz|ounces?)\b\.?",
    re.IGNORECASE,
)
_APOSTROPHE_RE = re.compile(r"['\u2018\u2019`]")
_OTHER_PUNCT_RE = re.compile(r"[^\w\s,]")
_COMMA_RE = re.compile(r"\s*,\s*")
_SPACE_RE = re.compile(r"\s+")
_MIXED_RE = re.compile(r",|\band\b|\bwith\b|&")


def is_mixed_dish(name: str) -> bool:
    """A name listing several components ("pork with carrot", "lotus nut, red bean and ...")."""
    return bool(_MIXED_RE.search(name.lower()))


def normalize_food_name(name: str | None) -> str:
    """Lowercase, unify hyphens/punctuation/whitespace and drop serving amounts.

    " Dry-Fried   Pomfret " -> "dry fried pomfret"
    "Orange 60 g" -> "orange"
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).lower()
    s = _AMOUNT_RE.sub(" ", s)
    s = _DASH_RE.sub(" ", s)
    s = _APOSTROPHE_RE.sub("", s)
    s = s.replace("&", " and ")
    s = _OTHER_PUNCT_RE.sub(" ", s)
    s = s.replace("_", " ")
    s = _COMMA_RE.sub(", ", s)
    s = _SPACE_RE.sub(" ", s)
    return s.strip(" ,")
