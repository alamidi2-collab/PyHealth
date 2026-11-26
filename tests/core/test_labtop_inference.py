"""Tiny offline regression tests for LabTOP prompt boundaries and task integration.

Every test uses at most two synthetic patients and a local tokenizer vocabulary;
no clinical dataset, pretrained weights, or network download is required.
"""

import math
from pathlib import Path

import polars as pl
import pytest
import torch
from transformers import BertTokenizerFast

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import BaseTask, LabTOPProcessor, LabTOPTask
from pyhealth.tasks.labtop import END, collate_labtop, collate_labtop_prompts


@pytest.fixture
def tokenizer(tmp_path: Path):
    """Create a real offline BERT tokenizer with a tiny synthetic vocabulary."""
    vocab = "[PAD] [UNK] [CLS] [SEP] [MASK] lab glucose sodium mg mmol / dl ."
    vocab += " " + " ".join(str(i) for i in range(20))
    vocab += " " + " ".join(f"##{i}" for i in range(10))
    (tmp_path / "vocab.txt").write_text("\n".join(vocab.split()))
    return BertTokenizerFast.from_pretrained(tmp_path, local_files_only=True)


def patient(labs, patient_id="p1", end=600):
    """Make one synthetic eICU patient; lab tuples are offset/item/value/unit."""
    rows = [
        {
            "event_type": "patient",
            "patient/age": "40",
            "patient/unitadmittime24": "00:00:00",
            "patient/unitdischargeoffset": str(end),
            "patient/patientunitstayid": patient_id,
        }
    ]
    for offset, item, value, unit in labs:
        rows.append(
            {
                "event_type": "lab",
                "lab/patientunitstayid": patient_id,
                "lab/labresultoffset": str(offset),
                "lab/labname": item,
                "lab/labresult": str(value),
                "lab/labmeasurenameinterface": unit,
            }
        )
    frame = pl.from_dicts(rows).with_columns(
        patient_id=pl.lit(patient_id), timestamp=pl.lit(None)
    )
    return Patient(patient_id, frame)


def test_prompt_excludes_entire_target_bin_and_future(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=128, max_days=2)
    p = patient(
        [
            (0, "sodium", 1, "mg"),
            (20, "glucose", 2, "mg"),
            (23, "glucose", 3, "mg"),
            (40, "glucose", 4, "mg"),
        ]
    )
    rows = list(processor.make_inference_dataset(p, max_new_tokens=20))
    assert [row["lab_value"] for row in rows] == [2, 3, 4]
    assert rows[0]["prompt_tokens"] == rows[1]["prompt_tokens"]
    initial = processor.preprocess(p)[0]["tokens"][:8]
    assert rows[0]["prompt_tokens"] == initial
    assert math.isnan(rows[0]["previous_value"])
    assert math.isnan(rows[1]["mean_previous_values"])
    assert rows[2]["previous_value"] == 3
    assert rows[2]["mean_previous_values"] == 2.5
    prefix, labels, _, _ = collate_labtop_prompts(rows[:2])
    assert prefix[0, -1] == tokenizer.convert_tokens_to_ids("glucose")
    assert labels[0, 0] == tokenizer.convert_tokens_to_ids("2")
    assert tokenizer.convert_tokens_to_ids("4") not in prefix[0]


def test_baselines_do_not_mix_units(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=128, max_days=2)
    p = patient(
        [(0, "glucose", 1, "mg"), (20, "glucose", 2, "mmol"), (40, "glucose", 3, "mg")]
    )
    first, second = processor.make_inference_dataset(p)
    assert first["previous_value_tokens"] == []
    assert math.isnan(first["previous_value"])
    assert math.isnan(first["mean_previous_values"])
    assert second["previous_value"] == 1
    assert second["mean_previous_values"] == 1


def test_mean_uses_unrounded_history_and_excludes_target(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=128, max_days=2)
    p = patient(
        [
            (0, "glucose", 0.001, "mg"),
            (20, "glucose", 0.002, "mg"),
            (40, "glucose", 0.003, "mg"),
            (60, "glucose", 19, "mg"),
        ]
    )
    final = list(processor.make_inference_dataset(p))[-1]
    assert final["mean_previous_values"] == pytest.approx(0.002)
    assert final["previous_value"] == 0.003


def test_context_is_whole_events_and_reserves_requested_generation(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=40, max_days=2)
    p = patient([(20 * i, "glucose", i, "mg") for i in range(8)])
    row = list(processor.make_inference_dataset(p, max_new_tokens=8))[-1]
    prefix, _, _, _ = collate_labtop_prompts([row])
    assert prefix.shape[1] + 8 <= 40
    assert row["prompt_tokens"][0] in processor.day_ids
    assert row["prompt_tokens"][-1] == tokenizer.convert_tokens_to_ids(END)
    # A historical event contains exactly 8 tokens with this vocabulary.
    assert len(row["prompt_tokens"]) % 8 == 0
    # Baselines still summarize all earlier same-unit measurements.
    assert row["mean_previous_values"] == 3


def test_target_length_does_not_change_selected_history(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=40, max_days=2)
    history = [(0, "glucose", 1, "mg"), (20, "glucose", 2, "mg")]
    a = patient(history + [(40, "glucose", 3, "mg")])
    b = patient(history + [(40, "glucose", 0.123456789, "mg")])
    first = list(processor.make_inference_dataset(a, max_new_tokens=8))[-1]
    second = list(processor.make_inference_dataset(b, max_new_tokens=8))[-1]
    assert first["prompt_tokens"] == second["prompt_tokens"]
    assert first["value_start"] == second["value_start"]


def test_no_history_short_stays_and_outside_stay_are_skipped(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=64, max_days=2)
    p = patient([(0, "glucose", 1, "mg"), (601, "glucose", 2, "mg")])
    assert list(processor.make_inference_dataset(p)) == []
    short = patient([(0, "glucose", 1, "mg"), (20, "glucose", 2, "mg")], end=30)
    assert processor.preprocess(short) == []
    assert list(processor.make_inference_dataset(short)) == []


@pytest.mark.parametrize("budget", [0, -1, 64, 1.5, True])
def test_invalid_generation_budget_rejected(tokenizer, budget):
    processor = LabTOPProcessor(tokenizer, max_seq_len=64, max_days=2)
    p = patient([(0, "glucose", 1, "mg")])
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(processor.make_inference_dataset(p, max_new_tokens=budget))


def test_task_adapter_and_sample_dataset_preserve_raw_ids(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=64, max_days=2)
    p = patient([(0, "glucose", 1, "mg"), (20, "glucose", 2, "mg")])
    with pytest.raises(ValueError, match="training"):
        LabTOPTask(processor)
    processor.fit([p])
    task = LabTOPTask(processor, lab_only=True)
    assert isinstance(task, BaseTask)
    rows = task(p)
    assert rows == processor.preprocess(p, lab_only=True)
    dataset = create_sample_dataset(rows, task.input_schema, task.output_schema)
    ids, types, positions = collate_labtop([dataset[0]])
    assert ids.dtype == types.dtype == positions.dtype == torch.long
    assert ids[0].tolist() == rows[0]["tokens"]
    assert types[0].tolist() == rows[0]["types"]


def test_item_selection_comes_only_from_fit_patients(tokenizer):
    processor = LabTOPProcessor(tokenizer, max_seq_len=64, max_days=2)
    train = patient([(0, "glucose", 1, "mg")], patient_id="train")
    held_out = patient([(0, "sodium", 2, "mg")], patient_id="test")
    processor.fit([train])
    task = LabTOPTask(processor)
    before = processor.selected_items.copy()
    assert task(held_out) == []
    assert processor.selected_items == before == {("lab", "glucose")}


def test_empty_collation_has_clear_error():
    with pytest.raises(ValueError, match="empty"):
        collate_labtop([])
    with pytest.raises(ValueError, match="empty"):
        collate_labtop_prompts([])
