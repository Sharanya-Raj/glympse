import pytest

from src.units import convert_to_grams, normalize_unit


@pytest.mark.parametrize(
    "unit, expected",
    [
        ("g", "g"), ("gram", "g"), ("grams", "g"), ("G", "g"), (" g ", "g"),
        ("kg", "kg"), ("ml", "ml"), ("mL", "ml"), ("ML", "ml"), ("L", "l"), ("l", "l"), ("litre", "l"),
        ("oz", "oz"), ("fl oz", "fl oz"), ("fl. oz", "fl oz"),
        ("bowl", None), ("", None), (None, None),
    ],
)
def test_normalize_unit(unit, expected):
    assert normalize_unit(unit) == expected


def test_mass_units():
    assert convert_to_grams("rice", 150, "g") == (150, 1.0)
    assert convert_to_grams("rice", 150, "grams") == (150, 1.0)
    assert convert_to_grams("rice", 1.5, "kg") == (1500, 1.0)
    grams, conf = convert_to_grams("nuts", 1, "oz")
    assert grams == pytest.approx(28.35, abs=0.01) and conf == 1.0


def test_water_like_liquid_ml_is_close_to_grams():
    grams, conf = convert_to_grams("ginger juice", 100, "ml")
    assert grams == pytest.approx(100)
    assert conf >= 0.85


def test_milk_density():
    grams, conf = convert_to_grams("milk", 250, "mL")
    assert grams == pytest.approx(257.5)
    assert conf == pytest.approx(0.9)


def test_litres():
    grams, _ = convert_to_grams("water", 1, "L")
    assert grams == pytest.approx(1000)


def test_oil_is_less_dense_than_water():
    grams, _ = convert_to_grams("sesame oil", 10, "ml")
    assert grams == pytest.approx(9.2)


def test_unknown_food_volume_has_low_confidence():
    grams, conf = convert_to_grams("braised pork", 100, "ml")
    assert grams == pytest.approx(100)
    assert conf == pytest.approx(0.5)


@pytest.mark.parametrize("amount, unit", [(None, "g"), (100, None), (100, "bowl"), (-5, "g")])
def test_unconvertible_returns_none_without_crashing(amount, unit):
    assert convert_to_grams("rice", amount, unit) == (None, 0.0)
