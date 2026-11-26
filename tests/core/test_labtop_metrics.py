"""Small numerical checks for LabTOP parsing and item-weighted evaluation."""

import math

import numpy as np
import pytest

from pyhealth.metrics.labtop import labtop_metrics, parse_lab_value


@pytest.mark.parametrize(
    "text,unit,expected",
    [
        ("1.0 - 3.0 mg", "mg", 2.0),
        ("-4", "", -4.0),
        ("-4--2", "", -3.0),
        ("+2.5", "", 2.5),
        (".5", "", 0.5),
        ("1 e - 3 mmol/L", "MMOL/l", 0.001),
        ("1e-3-3e-3", "", 0.002),
        ("1 2 3 mg / dL", "mg/dl", 123.0),
        ("[PAD]1##.##5mg|endofevent|[PAD]", "mg", 1.5),
        ("1e308-1e308", "", 1e308),
    ],
)
def test_parse_complete_numeric_field(text, unit, expected):
    assert parse_lab_value(text, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text,unit",
    [
        ("unknown", ""),
        ("", ""),
        ("NaN", ""),
        ("inf", ""),
        ("1e309", ""),
        ("1-1e309", ""),
        ("<5 mg", "mg"),
        ("5 mmol/l", "mg/dl"),
        ("result is 5 mg", "mg"),
        ("5 mg extra", "mg"),
        ("1/2", ""),
        ("5-6-7", ""),
        (None, ""),
    ],
)
def test_reject_invalid_unknown_or_qualified_predictions(text, unit):
    assert parse_lab_value(text, unit) is None


def test_independent_weighted_calculation_uses_percentile_span():
    """Three retained A values and one B value must receive weights 3:1."""
    records = [("a", value, value + 1) for value in range(5)]
    records += [("b", value, 0) for value in (10, 20, 30)]
    result = labtop_metrics(records, return_per_item=True)
    # A limits are 0.04, 3.96, retaining 1,2,3. B limits are 10.2,29.8,
    # retaining 20. These independent values detect retained-min/max mistakes.
    expected_nmae = (3 / 3.92 + 20 / 19.6) / 4
    expected_smape = (200 / 3 + 200 / 5 + 200 / 7 + 200) / 4
    assert result["weighted_nmae"] == pytest.approx(expected_nmae)
    assert result["weighted_smape"] == pytest.approx(expected_smape)
    assert result["retained"] == result["nmae_retained"] == 4
    assert result["trimmed"] == 4
    first, second = result["per_item"]
    assert first["lower_percentile"] == pytest.approx(0.04)
    assert first["upper_percentile"] == pytest.approx(3.96)
    assert first["retained"] == 3
    assert second["retained"] == 1


def test_percentiles_follow_dropping_unparseable_pairs():
    records = [
        ("a", 0, 0),
        ("a", 50, 51),
        ("a", 100, None),
        ("a", 150, 149),
        ("a", 200, 200),
    ]
    result = labtop_metrics(records, return_per_item=True)
    assert result["numeric_predictions"] == 4
    assert result["invalid_predictions"] == 1
    assert result["retained"] == 2
    # Percentiles of [0,50,150,200] are 1.5 and 198.5, span 197.
    assert result["weighted_nmae"] == pytest.approx(1 / 197)
    assert result["per_item"][0]["upper_percentile"] == pytest.approx(198.5)


def test_inclusive_limits_and_zero_zero_smape():
    records = [("a", value, value) for value in (0, 0, 0, 10, 10, 10)]
    result = labtop_metrics(records)
    assert result["retained"] == 6
    assert result["weighted_nmae"] == 0
    assert result["weighted_smape"] == 0


def test_constant_items_have_separate_explicit_nmae_denominator():
    records = [("a", value, value + 1) for value in range(5)]
    records += [("constant", 0, 0), ("constant", 0, 1)]
    result = labtop_metrics(records, return_per_item=True)
    assert result["retained"] == 5
    assert result["nmae_retained"] == 3
    assert result["constant_items"] == 1
    assert result["weighted_nmae"] == pytest.approx(1 / 3.92)
    assert result["weighted_smape"] == pytest.approx(
        (200 / 3 + 200 / 5 + 200 / 7 + 200) / 5
    )
    constant = result["per_item"][1]
    assert constant["status"] == "constant_truth"
    assert math.isnan(constant["nmae"])
    assert constant["smape"] == 100


def test_sparse_item_can_be_empty_after_trimming():
    result = labtop_metrics([("a", 0, 1), ("a", 10, 9)], return_per_item=True)
    assert result["numeric_predictions"] == 2
    assert result["retained"] == 0
    assert result["empty_after_trim_items"] == 1
    assert math.isnan(result["weighted_nmae"])
    assert math.isnan(result["weighted_smape"])
    assert result["per_item"][0]["status"] == "empty_after_trim"


def test_invalid_values_are_counted_without_coercing_strings():
    records = [
        ("a", float("nan"), 2),
        ("a", 2, float("inf")),
        ("a", "2", 3),
        ("a", 2, "3"),
        ("a", np.float64(2), np.float64(3)),
        ("b", 1, None),
    ]
    result = labtop_metrics(iter(records), return_per_item=True)
    assert result["prompts"] == 6
    assert result["numeric_predictions"] == 1
    assert result["invalid_truths"] == 2
    assert result["invalid_predictions"] == 3
    assert result["items"] == 2
    assert result["numeric_items"] == 1
    assert result["per_item"][1]["status"] == "no_numeric_pairs"


def test_negative_values_use_absolute_smape_denominator():
    result = labtop_metrics([("a", -1, 1), ("a", -2, -2)], percentiles=(0, 100))
    assert result["weighted_nmae"] == 1
    assert result["weighted_smape"] == 100


def test_empty_input_is_undefined_not_perfect():
    result = labtop_metrics([], return_per_item=True)
    assert result["prompts"] == result["retained"] == 0
    assert result["per_item"] == []
    assert math.isnan(result["weighted_nmae"])
    assert math.isnan(result["weighted_smape"])


@pytest.mark.parametrize("bounds", [(-1, 99), (1, 101), (50, 50), (99, 1)])
def test_invalid_percentiles_are_rejected(bounds):
    with pytest.raises(ValueError, match="percentiles"):
        labtop_metrics([], percentiles=bounds)
