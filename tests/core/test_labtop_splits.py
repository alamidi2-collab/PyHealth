"""Offline split-manifest validation using at most five invented patient IDs."""

import json

import pytest

from examples.labtop.splits import load_patient_splits

_IDS = ["synthetic_a", "synthetic_b", "synthetic_c", "synthetic_d"]


def _write(tmp_path, value):
    path = tmp_path / "local_splits.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _valid():
    return {
        "train": ["synthetic_b", "synthetic_a"],
        "valid": ["synthetic_c"],
        "test": ["synthetic_d"],
    }


def test_valid_splits_are_canonical_and_preserve_input_file(tmp_path):
    path = _write(tmp_path, _valid())
    before = path.read_bytes()
    result = load_patient_splits(path, iter(reversed(_IDS)))
    assert list(result) == ["train", "valid", "test"]
    assert result == {
        "train": ["synthetic_a", "synthetic_b"],
        "valid": ["synthetic_c"],
        "test": ["synthetic_d"],
    }
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content,match",
    [
        (
            {
                "train": ["synthetic_a"],
                "valid": ["synthetic_a"],
                "test": ["synthetic_c"],
            },
            "overlap",
        ),
        (
            {
                "train": ["synthetic_a", "synthetic_a"],
                "valid": ["synthetic_b"],
                "test": ["synthetic_c"],
            },
            "duplicate",
        ),
        (
            {
                "train": ["synthetic_a"],
                "valid": ["synthetic_b"],
                "test": ["synthetic_unknown"],
            },
            "unknown",
        ),
        (
            {
                "train": ["synthetic_a"],
                "valid": ["synthetic_b"],
                "test": ["synthetic_c"],
            },
            "unassigned",
        ),
        ({"train": [], "valid": ["synthetic_b"], "test": ["synthetic_c"]}, "nonempty"),
        (
            {"train": "synthetic_a", "valid": ["synthetic_b"], "test": ["synthetic_c"]},
            "nonempty",
        ),
        ({"train": [1], "valid": ["synthetic_b"], "test": ["synthetic_c"]}, "string"),
        (
            {"train": [None], "valid": ["synthetic_b"], "test": ["synthetic_c"]},
            "string",
        ),
        (
            {"train": ["  "], "valid": ["synthetic_b"], "test": ["synthetic_c"]},
            "nonblank",
        ),
        (
            {
                "train": ["synthetic_a"],
                "validation": ["synthetic_b"],
                "test": ["synthetic_c"],
            },
            "exactly",
        ),
        ({**_valid(), "extra": []}, "exactly"),
        (["train", "valid", "test"], "exactly"),
    ],
)
def test_invalid_memberships_fail_without_exposing_identifiers(
    tmp_path, content, match
):
    path = _write(tmp_path, content)
    with pytest.raises(ValueError, match=match) as error:
        load_patient_splits(path, _IDS)
    assert "synthetic_" not in str(error.value)


@pytest.mark.parametrize(
    "text",
    [
        '{"train": invalid}',
        (
            '{"train": ["synthetic_a"], "train": ["synthetic_b"], '
            '"valid": ["synthetic_c"], "test": ["synthetic_d"]}'
        ),
    ],
)
def test_invalid_json_and_duplicate_object_keys_are_rejected(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_patient_splits(path, _IDS)
    assert "synthetic_" not in str(error.value)


@pytest.mark.parametrize(
    "roster", [[], "synthetic_a", [1, "synthetic_b"], ["synthetic_a", "synthetic_a"]]
)
def test_invalid_dataset_roster_is_rejected(tmp_path, roster):
    path = _write(tmp_path, _valid())
    with pytest.raises(ValueError) as error:
        load_patient_splits(path, roster)
    assert "synthetic_" not in str(error.value)


def test_whitespace_is_not_silently_normalized_into_an_existing_id(tmp_path):
    content = _valid()
    content["test"] = [" synthetic_d "]
    with pytest.raises(ValueError, match="unknown"):
        load_patient_splits(_write(tmp_path, content), _IDS)
