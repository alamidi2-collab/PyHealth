"""Exercise the HiRID bridge, causal loss and held-out evaluation offline."""

import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch
from torch.nn import functional as F
from transformers import BertTokenizerFast

from pyhealth.data import Patient
from pyhealth.metrics.labtop import labtop_metrics, parse_lab_value
from pyhealth.models import LabTOP
from pyhealth.tasks.labtop import LabTOPHiRIDProcessor, collate_labtop, collate_labtop_prompts


class TestLabTOP(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.txt"
            vocab.write_text("\n".join(
                "[PAD] [UNK] [CLS] [SEP] [MASK] observation _ tables glucose sodium 1 2 3 mg".split()
            ))
            self.tokenizer = BertTokenizerFast.from_pretrained(directory)
            reference = Path(directory) / "reference.csv"
            reference.write_text("Source Table,ID,Variable Name,Unit\n"
                                 "Observation,1,glucose,mg\nObservation,2,sodium,mg\n")
            self.processor = LabTOPHiRIDProcessor(self.tokenizer, reference, max_seq_len=64, max_days=3)
        admission = datetime(2024, 1, 1, 17, 50)
        rows = [{"event_type": "general_table", "timestamp": admission}]
        for offset, code, value in [(0, 1, 1), (380, 1, 2), (383, 2, 3), (-1, 2, 2)]:
            rows.append({"event_type": "observation_tables",
                         "timestamp": admission + timedelta(minutes=offset),
                         "observation_tables/variableid": code, "observation_tables/value": value})
        frame = pl.from_dicts(rows).with_columns(patient_id=pl.lit("p"))
        self.patient = Patient("p", frame)

    def test_bridge_and_prompt_boundaries(self):
        self.processor.fit([self.patient])
        row, = self.processor.preprocess(self.patient)
        self.assertEqual(row["types"], ([0] * 8 + [1] * 3) * 3)
        short = {key: row[key][:11] for key in ("tokens", "types")}
        ids, masks, positions = collate_labtop([row, short])
        self.assertEqual(ids.shape, (2, 33))
        self.assertTrue(torch.equal(masks[1, 11:], torch.zeros(22)))
        self.assertEqual(positions[1].tolist(), list(range(11)) + [0] * 22)
        prompts = list(self.processor.make_inference_dataset(self.patient))
        self.assertEqual([r["lab_value"] for r in prompts], [2, 3])
        self.assertEqual([r["prompt_tokens"] for r in prompts], [short["tokens"]] * 2)
        prefix, truth, previous, means = collate_labtop_prompts(prompts)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(prefix[0, -8:].tolist()),
                         ["[DAY_2]", "[TUE]", "[00h]", "[10m]", "observation", "_", "tables", "glucose"])
        self.assertEqual(truth.shape, (2, 3))
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(truth[0].tolist()),
                         ["2", "mg", "|endofevent|"])
        self.assertEqual(previous[0].tolist(), short["tokens"][:-1])
        self.assertEqual(means[0], 1)
        self.assertTrue(math.isnan(means[1]))

    def test_shift_padding_and_backward(self):
        rows = self.processor.preprocess(self.patient)
        short = {key: rows[0][key][:11] for key in ("tokens", "types")}
        ids, mask, pos = self.processor.collate_train([*rows, short])
        model = LabTOP(self.tokenizer, 64, 8, 1, 2, dropout=0.0, lab_only=True)
        result = model(ids, mask, pos)
        self.assertEqual(result["logits"].shape, (2, 33, len(self.tokenizer)))
        keep = mask[:, 1:].bool()
        expected = F.cross_entropy(result["logits"][:, :-1][keep], ids[:, 1:][keep])
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        self.assertIsNotNone(model.model.transformer.wte.weight.grad)

    def test_numeric_metrics(self):
        self.assertEqual(parse_lab_value("1.0 - 3.0 mg", "mg"), 2)
        self.assertEqual(parse_lab_value("-4"), -4)
        self.assertIsNone(parse_lab_value("unknown"))
        records = [("a", x, x + 1) for x in range(5)] + [("b", 0, 0), ("a", 4, None)]
        result = labtop_metrics(records)
        self.assertEqual(result["retained"], 4)
        self.assertEqual(result["numeric_predictions"], 6)
        self.assertAlmostEqual(result["weighted_nmae"], 1 / 3.92)
        self.assertAlmostEqual(result["weighted_smape"], sum(200 / n for n in [3, 5, 7]) / 4)
