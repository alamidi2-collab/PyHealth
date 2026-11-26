"""Synthetic checks of split integrity, held-out evaluation and history ablation."""

import copy
import json
import math
import random
from itertools import pairwise

import pytest
import torch
from torch import nn

from examples.labtop.train import (
    ablate_history,
    evaluate,
    json_safe,
    load_model_weights,
    split_patient_ids,
    write_json,
)
from pyhealth.tasks.labtop import END

# Explicit re-export registers the existing pytest fixture in this test module.
from tests.core.test_labtop_model import tokenizer as tokenizer  # noqa: PLC0414


class FakeGenerator(nn.Module):
    """Record exactly which tokens an evaluator exposes to generation."""

    def __init__(self, outputs, failure=None):
        super().__init__()
        self.child = nn.Dropout()
        self.outputs = iter(outputs)
        self.failure = failure
        self.calls = []

    def predict_value_tokens(self, prompt_ids, end_token_id, max_new_tokens):
        assert not self.training
        assert not self.child.training
        assert end_token_id > 0
        assert max_new_tokens == 3
        self.calls.append(prompt_ids.tolist())
        if self.failure is not None:
            raise self.failure
        return next(self.outputs)


def make_rows(tokenizer):
    """Four targets with separate historical values and unpublished answers."""
    tokenizer.add_tokens(["4"])
    end = tokenizer.convert_tokens_to_ids(END)
    header = tokenizer.encode("lab glucose", add_special_tokens=False)
    history = tokenizer.encode("lab glucose 1 mg", add_special_tokens=False) + [end]
    rows = []
    for value in range(1, 5):
        rows.append(
            {
                "patient_id": "synthetic_patient",
                "icu_id": "synthetic_stay",
                "item": "glucose",
                "unit": "mg",
                "prompt_tokens": history.copy(),
                "lab_tokens": header
                + tokenizer.encode(f"{value} mg", add_special_tokens=False)
                + [end],
                "value_start": len(header),
                "lab_value": float(value),
                "previous_value_tokens": [] if value == 1 else history[:-1],
                "previous_value": None if value == 1 else float(value - 1),
                "mean_previous_values": math.nan if value == 1 else value / 2,
            }
        )
    return rows


def prediction_tokens(tokenizer):
    return [
        tokenizer.encode(text, add_special_tokens=False)
        for text in ("2 mg", "3 mg", "4 mg", "unknown")
    ]


def test_split_stability_disjointness_and_duplicate_patient_ids():
    ids = ["synthetic_c", "synthetic_a", "synthetic_b", "synthetic_e", "synthetic_d"]
    previous_rng_state = random.getstate()
    split = split_patient_ids(ids + [ids[0]], seed=42)
    assert split == split_patient_ids(list(reversed(ids)), seed=42)
    assert split != split_patient_ids(ids, seed=17)
    assert random.getstate() == previous_rng_state
    train, valid, test = (set(split[key]) for key in ("train", "valid", "test"))
    assert len(train) == 3 and len(valid) == len(test) == 1
    assert not (train & valid or train & test or valid & test)
    assert train | valid | test == set(ids)
    with pytest.raises(ValueError, match="three distinct"):
        split_patient_ids(["synthetic_a", "synthetic_a", "synthetic_b"])


def test_evaluation_uses_only_prefix_and_separates_missing_baselines(tokenizer):
    rows = make_rows(tokenizer)
    original = copy.deepcopy(rows)
    model = FakeGenerator(prediction_tokens(tokenizer))
    model.train()
    model.child.eval()
    modes = {module: module.training for module in model.modules()}
    result = evaluate(model, rows, tokenizer, 3, return_details=True)
    assert model.calls == [
        row["prompt_tokens"] + row["lab_tokens"][: row["value_start"]] for row in rows
    ]
    assert {module: module.training for module in model.modules()} == modes
    assert rows == original
    assert set(result) == {"model", "last_value", "history_mean"}
    for method in result.values():
        assert method["prompts"] == 4
        assert method["numeric_predictions"] == 3
        assert method["numeric_coverage"] == 0.75
        assert method["invalid_predictions"] == 1
    # The model's valid truths are 1,2,3; each baseline's are 2,3,4.
    # Their 1st/99th percentile spans are both 1.96, but subsets are distinct.
    assert result["model"]["weighted_nmae"] == pytest.approx(1 / 1.96)
    assert result["last_value"]["weighted_nmae"] == pytest.approx(1 / 1.96)
    assert result["history_mean"]["weighted_nmae"] == pytest.approx(1.5 / 1.96)
    assert result["model"]["per_item"][0]["lower_percentile"] == pytest.approx(1.02)
    assert result["last_value"]["per_item"][0]["lower_percentile"] == pytest.approx(
        2.02
    )
    assert "patient_id" not in json.dumps(json_safe(result))

    # Alter answers and their token lengths; generation must see identical inputs.
    altered = copy.deepcopy(rows)
    for row in altered:
        row["lab_tokens"][row["value_start"] :] = [1] * 9
        row["lab_value"] = 99.0
    other_model = FakeGenerator(prediction_tokens(tokenizer))
    evaluate(other_model, altered, tokenizer, 3)
    assert other_model.calls == model.calls


def test_evaluation_keeps_units_separate(tokenizer):
    rows = make_rows(tokenizer)[:2]
    rows[1]["unit"] = "units"
    tokenizer.add_tokens(["units"])
    model = FakeGenerator(
        [
            tokenizer.encode(text, add_special_tokens=False)
            for text in ("2 mg", "3 units")
        ]
    )
    result = evaluate(model, rows, tokenizer, 3, return_details=True)["model"]
    assert result["numeric_predictions"] == 2
    assert result["items"] == result["constant_items"] == 2
    assert {tuple(item["item"]) for item in result["per_item"]} == {
        ("glucose", "mg"),
        ("glucose", "units"),
    }


def test_empty_or_failed_evaluation_restores_modes(tokenizer):
    rows = make_rows(tokenizer)
    model = FakeGenerator([], failure=RuntimeError("inference failed"))
    model.train()
    model.child.eval()
    modes = {module: module.training for module in model.modules()}
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluate(model, rows, tokenizer, 3)
    assert {module: module.training for module in model.modules()} == modes
    with pytest.raises(ValueError, match="no eligible"):
        evaluate(model, [], tokenizer, 3)
    assert {module: module.training for module in model.modules()} == modes


def events_from_tokens(tokens, end):
    events, current = [], []
    for token in tokens:
        current.append(token)
        if token == end:
            events.append(tuple(current))
            current = []
    assert current == []
    return events


def test_history_ablation_removes_whole_nested_events_without_changing_target():
    end = 99
    events = [(index, index + 20, end) for index in range(1, 9)]
    row = {
        "prompt_tokens": [token for event in events for token in event],
        "lab_tokens": [30, 31, 77, end],
        "value_start": 2,
        "lab_value": 77.0,
        "unit": "invented_units",
        "previous_value": 76.0,
        "mean_previous_values": 73.5,
    }
    original = copy.deepcopy(row)
    outputs = [ablate_history(row, keep, 11, end) for keep in (0, 0.25, 0.75, 1)]
    retained = [events_from_tokens(output["prompt_tokens"], end) for output in outputs]
    assert retained[0] == []
    assert retained[-1] == events
    for fewer, more in pairwise(retained):
        assert set(fewer) <= set(more)
        assert fewer == [event for event in more if event in set(fewer)]
    for output, kept in zip(outputs, retained):
        assert output["history_events"] == 8
        assert output["retained_history_events"] == len(kept)
        for key in row.keys() - {"prompt_tokens"}:
            assert output[key] == row[key]
    assert row == original
    assert outputs[1] == ablate_history(row, 0.25, 11, end)


def test_history_ablation_rejects_incomplete_context_and_invalid_probabilities():
    with pytest.raises(ValueError, match="complete events"):
        ablate_history({"prompt_tokens": [1, 99, 2]}, 0.5, 1, 99)
    for probability in (-0.1, 1.1, math.nan):
        with pytest.raises(ValueError, match="probabilities"):
            ablate_history({"prompt_tokens": [1, 99]}, probability, 1, 99)


def test_json_null_for_undefined_metrics(tmp_path):
    source = {
        "score": math.nan,
        "nested": [math.inf, -math.inf, (1.0, True, None)],
        "path": tmp_path,
    }
    path = tmp_path / "metrics.json"
    write_json(path, source)
    encoded = path.read_text()
    assert "NaN" not in encoded and "Infinity" not in encoded
    result = json.loads(encoded)
    assert result["score"] is None
    assert result["nested"] == [None, None, [1.0, True, None]]
    assert result["path"] == str(tmp_path)
    assert math.isnan(source["score"])


class TinyCheckpointModel(nn.Module):
    """Exercise checkpoint validation without constructing a language model."""

    def __init__(self):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))
        self.model = nn.Linear(2, 1)


def test_legacy_checkpoint_allows_only_missing_base_model_marker():
    model = TinyCheckpointModel()
    old = {
        name: torch.full_like(value, 0.25)
        for name, value in model.state_dict().items()
        if name != "_dummy_param"
    }
    load_model_weights(model, {"model_state_dict": old, "model_config": {}})
    for name, value in old.items():
        assert torch.equal(model.state_dict()[name], value)
    load_model_weights(model, old)
    load_model_weights(model, model.state_dict())


def test_invalid_checkpoint_cannot_silently_leave_random_model_weights():
    model = TinyCheckpointModel()
    original = model.state_dict()
    missing = {name: value for name, value in original.items() if name != "model.bias"}
    with pytest.raises(ValueError, match="missing"):
        load_model_weights(model, missing)
    unexpected = {**original, "unknown.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="unexpected"):
        load_model_weights(model, unexpected)
    wrong_shape = {**original, "model.weight": torch.zeros(3, 3)}
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model_weights(model, wrong_shape)
