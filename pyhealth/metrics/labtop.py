"""Item-wise trimmed errors and strict numeric parsing for LabTOP.

The defined-case calculation follows the authors' ``PerformanceEvaluator``:
drop nonnumeric pairs, find each item's 1st/99th truth percentiles, retain
inclusive bounds, divide MAE by that percentile span, and weight item errors by
the retained counts. SMAPE is a percentage in [0, 200], not a fraction.

Reference: https://github.com/sujeongim/LabTOP
Reviewed commit: fa2e66749af84338f6cef7ba3c5918e09719c0e4
Files: ``labtop/src/core/utils/metrics.py`` and ``post_processor.py``.
Paper: https://proceedings.mlr.press/v287/im25a.html

Two explicit robustness choices differ from undefined/fragile reference cases:
nonfinite pairs are rejected, and constant items contribute only to SMAPE.
Empty trimmed groups are counted, not assigned zero error. The parser preserves
signed integers and scientific notation and rejects arbitrary surrounding text;
the reference regex can lose integer signs or mistake a range dash for a sign.
These fixes do not establish reproduction of the paper's clinical results.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Hashable, Iterable
from numbers import Real
from typing import Any

import numpy as np

__all__ = ["labtop_metrics", "parse_lab_value"]

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_VALUE_OR_RANGE = re.compile(rf"({_NUMBER})(?:-({_NUMBER}))?")


def parse_lab_value(text: str, unit: str = "") -> float | None:
    """Parse a finite number or a numeric range midpoint from decoded tokens.

    Spaces are removed because LabTOP uses digit-wise tokenization. WordPiece
    continuation markers, padding, and one final end-of-event marker are removed
    as decoding artifacts. Only the supplied unit may be removed, and only as a
    suffix; this never extracts a number from an arbitrary clinical sentence.
    Signed integers, decimals, and scientific notation are supported. Inequality
    qualifiers such as ``<5`` are not exact numbers and remain unparseable.

    Args:
        text: Decoded numeric-value tokens, optionally followed by a unit.
            A complete event containing timestamps or item names is not valid.
        unit: Expected unit suffix, matched without case or whitespace.

    Returns:
        A finite value, the midpoint of two finite range endpoints, or ``None``
        for an invalid expression, unrecognized unit, or numerical overflow.

    Examples:
        >>> parse_lab_value("1.0 - 3.0 mg", "mg")
        2.0
        >>> parse_lab_value("-4")
        -4.0
        >>> parse_lab_value("1 e - 3 mmol/L", "mmol/l")
        0.001
    """
    if not isinstance(text, str) or not isinstance(unit, str):
        return None
    text = "".join(text.lower().split()).replace("##", "").replace("[pad]", "")
    text = text.removesuffix("|endofevent|")
    unit = "".join(unit.lower().split())
    if unit and text.endswith(unit):
        text = text[: -len(unit)]
    match = _VALUE_OR_RANGE.fullmatch(text)
    if match is None:
        return None
    lower = float(match[1])
    upper = lower if match[2] is None else float(match[2])
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    # Halve first so two large, finite endpoints do not overflow on addition.
    return lower if match[2] is None else lower / 2 + upper / 2


def _finite_number(value: Any) -> bool:
    """Recognize finite real scalar values without coercing failed text."""
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def labtop_metrics(
    records: Iterable[tuple[Hashable, float, float | None]],
    *,
    percentiles: tuple[float, float] = (1.0, 99.0),
    return_per_item: bool = False,
) -> dict[str, Any]:
    """Aggregate LabTOP errors from ``(item, truth, prediction)`` triples.

    For each item, form the finite numeric truth/prediction pairs first, then
    compute the requested truth percentiles using linear interpolation. Retain
    truths within both inclusive limits. NMAE is mean absolute error divided by
    the *percentile span*, not the min/max range of the retained observations.
    SMAPE is ``200 * abs(truth - prediction) / (abs(truth) + abs(prediction))``;
    a zero/zero pair has zero error. Aggregate by retained observation counts,
    so this is not an unweighted average across laboratory items.

    Constant items have undefined NMAE and are excluded from its numerator and
    denominator, while still contributing to SMAPE. This explicitly preserves
    the group's robust policy instead of reproducing the reference's undefined
    division by zero. An item with two distinct observations may retain neither
    after 1st/99th-percentile trimming. Such groups are reported and contribute
    to neither metric. Empty aggregate denominators return ``NaN``.

    Parse failures are not scored as zeros or silently imputed. As in the
    reference, dropping failed pairs can change an item's percentile limits.
    Report coverage alongside scores. Evaluate model, previous-value, and
    history-mean predictions in separate calls; their valid subsets may differ.

    Args:
        records: Finite numeric observations and predictions, with ``None`` for
            failed predictions. Nonnumeric/nonfinite values are counted and
            dropped. Keys should distinguish laboratory items and incompatible
            units; no unit conversion is performed by this function.
        percentiles: Lower and upper truth percentiles. The reference uses
            ``(1, 99)``. Alternative values change the evaluation protocol.
        return_per_item: Include item counts, limits, and errors when true.

    Returns:
        Aggregate scores and coverage counts. ``numeric_predictions`` counts
        finite numeric pairs, ``retained`` is the SMAPE denominator, and
        ``nmae_retained`` is the NMAE denominator. ``per_item`` is added only
        when requested. Invalid truth/prediction counts can overlap.

    Raises:
        ValueError: If the percentile bounds do not satisfy
            ``0 <= lower < upper <= 100``.
    """
    if (
        len(percentiles) != 2
        or not all(_finite_number(value) for value in percentiles)
        or not 0 <= percentiles[0] < percentiles[1] <= 100
    ):
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    groups: dict[Hashable, list[tuple[float, float]]] = defaultdict(list)
    item_totals: dict[Hashable, int] = defaultdict(int)
    total = numeric = invalid_truths = invalid_predictions = 0
    for item, truth, prediction in records:
        total += 1
        item_totals[item] += 1
        truth_ok, pred_ok = _finite_number(truth), _finite_number(prediction)
        invalid_truths += int(not truth_ok)
        invalid_predictions += int(not pred_ok)
        if truth_ok and pred_ok:
            groups[item].append((float(truth), float(prediction)))
            numeric += 1

    nmae_sum = smape_sum = 0.0
    retained = normalized_count = constant_items = empty_items = 0
    retained_items = 0
    details = []
    for item, item_total in item_totals.items():
        pairs = groups.get(item, [])
        result = {
            "item": item,
            "prompts": item_total,
            "numeric_predictions": len(pairs),
            "retained": 0,
            "nmae_retained": 0,
            "lower_percentile": math.nan,
            "upper_percentile": math.nan,
            "mae": math.nan,
            "nmae": math.nan,
            "smape": math.nan,
            "status": "no_numeric_pairs",
        }
        if not pairs:
            details.append(result)
            continue
        truth, prediction = np.asarray(pairs, dtype=np.float64).T
        low, high = np.percentile(truth, percentiles, method="linear")
        result.update(lower_percentile=float(low), upper_percentile=float(high))
        keep = (truth >= low) & (truth <= high)
        truth, prediction = truth[keep], prediction[keep]
        count = len(truth)
        if count == 0:
            empty_items += 1
            result["status"] = "empty_after_trim"
            details.append(result)
            continue
        error = np.abs(truth - prediction)
        denominator = (np.abs(truth) + np.abs(prediction)) / 2
        percentages = np.divide(
            100 * error,
            denominator,
            out=np.zeros_like(error),
            where=denominator != 0,
        )
        mae, smape = float(error.mean()), float(percentages.mean())
        smape_sum += float(percentages.sum())
        retained += count
        retained_items += 1
        result.update(retained=count, mae=mae, smape=smape, status="retained")
        if high > low:
            nmae = mae / float(high - low)
            nmae_sum += nmae * count
            normalized_count += count
            result.update(nmae=nmae, nmae_retained=count)
        else:
            constant_items += 1
            result["status"] = "constant_truth"
        details.append(result)

    metrics = {
        "weighted_nmae": nmae_sum / normalized_count if normalized_count else math.nan,
        "weighted_smape": smape_sum / retained if retained else math.nan,
        "prompts": total,
        "numeric_predictions": numeric,
        "retained": retained,
        "nmae_retained": normalized_count,
        "invalid_truths": invalid_truths,
        "invalid_predictions": invalid_predictions,
        "trimmed": numeric - retained,
        "items": len(item_totals),
        "numeric_items": len(groups),
        "retained_items": retained_items,
        "constant_items": constant_items,
        "empty_after_trim_items": empty_items,
        "percentiles": [float(value) for value in percentiles],
    }
    if return_per_item:
        metrics["per_item"] = details
    return metrics
