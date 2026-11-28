"""Item-wise trimmed errors for LabTOP numeric predictions."""

from collections import defaultdict
import re

import numpy as np
from scipy.stats import spearmanr


def _value_text(text, unit):
    text = "".join(text.lower().split())
    unit = "".join(unit.lower().split())
    return text.removesuffix(unit) if unit else text


def parse_lab_value(text, unit=""):
    """Parse a generated number or the midpoint of a numeric range."""
    text = _value_text(text, unit)
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
    match = re.fullmatch(rf"({number})(?:-({number}))?", text)
    if match:
        lower = float(match[1])
        return lower if match[2] is None else (lower + float(match[2])) / 2
    return None


def labtop_metrics(records):
    """Report LabTOP's per-item errors and their count-weighted averages.

    Percentiles are computed on numeric pairs before trimming. Items with a
    zero percentile span contribute to SMAPE only.
    """
    groups = defaultdict(list)
    total = numeric = 0
    for item, truth, pred in records:
        total += 1
        pairs = groups[item]
        if pred is not None and np.isfinite(pred) and np.isfinite(truth):
            pairs.append((truth, pred))
            numeric += 1

    per_item = []
    for item, pairs in groups.items():
        low = high = mae = nmae = smape = np.nan
        count = 0
        if pairs:
            truth, pred = np.asarray(pairs, dtype=float).T
            low, high = np.percentile(truth, [1, 99])
            # LabTOP trims by true value before averaging. With two distinct
            # truths, both fall outside these bounds, even for perfect predictions.
            # A retained perfect prediction contributes zero error and one count.
            keep = (truth >= low) & (truth <= high)
            truth, pred = truth[keep], pred[keep]
            count = len(truth)
            if count:
                error = np.abs(truth - pred)
                mae = error.mean()
                if high > low:
                    nmae = mae / (high - low)
                denominator = (np.abs(truth) + np.abs(pred)) / 2
                smape = np.divide(100 * error, denominator, out=np.zeros_like(error),
                                  where=denominator != 0).mean()
        per_item.append(dict(item=item, count=count, min_v=float(low), max_v=float(high),
                             mae=float(mae), nmae=float(nmae), smape=float(smape)))
    retained = sum(row["count"] for row in per_item)

    def weighted(metric):
        eligible = [row for row in per_item if row["count"] and not np.isnan(row[metric])]
        count = sum(row["count"] for row in eligible)
        if not count:
            return np.nan
        return sum(row["count"] * row[metric] for row in eligible) / count

    return {"weighted_nmae": weighted("nmae"), "weighted_smape": weighted("smape"),
            "prompts": total, "numeric_predictions": numeric, "retained": retained,
            "per_item": per_item}

def mean_value_token_entropy(tokenizer, ids, token_entropies, text, unit):
    """Mean entropy for the generated tokens overlapping the parsed value text."""
    value = _value_text(text, unit)
    # Decode the original WordPiece prefixes: re-encoding can change segmentation.
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end]).replace(" ##", "")
        if len("".join(prefix.split())) >= len(value):
            return float(np.mean(token_entropies[:end]))
    raise ValueError("Generated tokens did not cover the parsed value")


def labtop_uncertainty(records, uncertainties, per_item):
    """Pool non-perfect, parseable predictions before trimming, scaled by item span."""
    spans = {row["item"]: row["max_v"] - row["min_v"] for row in per_item}
    errors, scores = [], []
    for (item, truth, pred), uncertainty_score in zip(records, uncertainties, strict=True):
        if uncertainty_score is None or not np.isfinite(uncertainty_score) or pred is None or not np.isfinite(pred):
            continue
        span = spans[item]
        if np.isfinite(truth) and truth != pred and span > 0:
            errors.append(abs(truth - pred) / span)
            scores.append(uncertainty_score)
    rho = np.nan
    if len(set(scores)) > 1 and len(set(errors)) > 1:
        rho = float(spearmanr(scores, errors).statistic)
    return {"spearman_rho": rho, "sample_count": len(scores)}
