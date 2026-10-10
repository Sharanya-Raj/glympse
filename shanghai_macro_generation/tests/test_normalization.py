import pytest

from src.normalize_food import normalize_food_name


@pytest.mark.parametrize("raw", ["Orange", "orange ", "ORANGE", "  Orange\t"])
def test_case_and_whitespace(raw):
    assert normalize_food_name(raw) == "orange"


@pytest.mark.parametrize(
    "raw", ["Dry fried pomfret", "dry-fried pomfret", "Dry Fried Pomfret", " Dry-Fried   Pomfret ", "Dry\u2013fried pomfret"]
)
def test_hyphens_collapse_to_same_name(raw):
    assert normalize_food_name(raw) == "dry fried pomfret"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Orange 60 g", "orange"),
        ("Ginger juice 100 ml", "ginger juice"),
        ("Tomato 96.2 g", "tomato"),
        ("Rice 1.5kg", "rice"),
    ],
)
def test_serving_amounts_removed(raw, expected):
    assert normalize_food_name(raw) == expected


def test_commas_kept_and_spaced():
    assert (
        normalize_food_name("Lotus nut,red bean , poria coco and orange peel 30 g")
        == "lotus nut, red bean, poria coco and orange peel"
    )


def test_ampersand_and_punctuation():
    assert normalize_food_name("Pork & cabbage (stir-fried)!") == "pork and cabbage stir fried"


def test_does_not_merge_different_foods():
    assert normalize_food_name("Rice") != normalize_food_name("Fried rice")
    assert normalize_food_name("Egg") != normalize_food_name("Steamed egg")


def test_empty():
    assert normalize_food_name("") == ""
    assert normalize_food_name(None) == ""
