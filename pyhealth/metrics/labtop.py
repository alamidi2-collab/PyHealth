"""Item-wise trimmed errors for LabTOP numeric predictions."""

from collections import defaultdict
import re

import numpy as np


def parse_lab_value(text, unit=""):
    """Parse a generated number or the midpoint of a numeric range."""
    text = "".join(text.lower().split())
    unit = "".join(unit.lower().split())
    if unit and text.endswith(unit):
        text = text[:-len(unit)]
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
    match = re.fullmatch(rf"({number})(?:-({number}))?", text)
    if match:
        lower = float(match[1])
        return lower if match[2] is None else (lower + float(match[2])) / 2
    return None


def labtop_metrics(records):
    """Aggregate (item, truth, prediction) triples using retained item counts.

    Trim outside each item's inclusive 1st and 99th truth percentiles after
    dropping unparseable predictions. A constant item contributes to SMAPE
    only, since its normalization range is zero. Undefined means are NaN.
    """
    groups = defaultdict(list)
    total = numeric = 0
    for item, truth, pred in records:
        total += 1
        if pred is not None and np.isfinite(pred) and np.isfinite(truth):
            groups[item].append((truth, pred))
            numeric += 1
    nmae = smape = 0.0
    count = normalized_count = 0
    for pairs in groups.values():
        truth, pred = np.asarray(pairs, dtype=float).T
        low, high = np.percentile(truth, [1, 99])
        keep = (truth >= low) & (truth <= high)
        truth, pred = truth[keep], pred[keep]
        error = np.abs(truth - pred)
        denominator = (np.abs(truth) + np.abs(pred)) / 2
        smape += np.divide(100 * error, denominator, out=np.zeros_like(error),
                           where=denominator != 0).sum()
        count += len(truth)
        if high > low:
            nmae += error.sum() / (high - low)
            normalized_count += len(truth)
    return {"weighted_nmae": nmae / normalized_count if normalized_count else np.nan,
            "weighted_smape": smape / count if count else np.nan,
            "prompts": total, "numeric_predictions": numeric, "retained": count}
