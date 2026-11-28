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
from pyhealth.metrics.labtop import labtop_metrics, labtop_uncertainty, parse_lab_value, mean_value_token_entropy
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

    def test_generated_entropy(self):
        model = LabTOP(self.tokenizer, 64, 8, 1, 2, dropout=0.0).eval()
        prompt = self.tokenizer.encode("glucose", add_special_tokens=False)
        # Compare the captured distributions with independent forward passes.
        end = self.tokenizer.convert_tokens_to_ids("|endofevent|")
        with torch.no_grad():
            expected = model.predict_value_tokens(prompt, end, 3)
            ids, entropies = model.predict_value_tokens(prompt, end, 3, return_entropies=True)
            self.assertEqual(ids, expected)
            self.assertEqual(len(ids), len(entropies))
            for step, (token, entropy) in enumerate(zip(ids, entropies)):
                logits = model.model(torch.tensor([prompt + ids[:step]])).logits[0, -1]
                self.assertEqual(token, logits.argmax().item())
                self.assertAlmostEqual(entropy, torch.distributions.Categorical(logits=logits).entropy().item(), places=5)
        tokens = self.tokenizer.encode("1 2 mg", add_special_tokens=False)
        self.assertEqual(parse_lab_value(self.tokenizer.decode(tokens), "mg"), 12)
        self.assertEqual(mean_value_token_entropy(self.tokenizer, tokens, [2, 4, 100], "1 2 mg", "mg"), 3)

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
        a, b = result["per_item"]
        self.assertEqual((a["item"], a["count"], a["mae"]), ("a", 3, 1))
        self.assertAlmostEqual(a["nmae"], 1 / 3.92)
        self.assertTrue(math.isnan(b["nmae"]))
        self.assertEqual(b["smape"], 0)
        result = labtop_metrics([("constant", 2, 3), ("unparsed", 1, None)])
        self.assertTrue(math.isnan(result["weighted_nmae"]))
        self.assertEqual(result["per_item"][1]["count"], 0)


class TestLabTOPReporting(unittest.TestCase):
    def test_pooled_uncertainty(self):
        records = [("a", 0, 2), ("a", 10, 14), ("b", 0, 10), ("b", 100, 100),
                   ("a", 5, None), ("constant", 3, 1)]
        report = labtop_metrics(records)
        # Even the extrema trimmed from MAE participate. The largest raw error
        # belongs to b and becomes the smallest after normalizing by its span.
        result = labtop_uncertainty(records, [2, 3, 1, 9, None, 7], report["per_item"])
        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["spearman_rho"], 1)
        tied = labtop_uncertainty(records, [1] * len(records), report["per_item"])
        self.assertTrue(math.isnan(tied["spearman_rho"]))

    def test_absent_item(self):
        report = labtop_metrics([("heart rate", 70, 70)])
        self.assertEqual([row["item"] for row in report["per_item"]], ["heart rate"])
        empty = labtop_metrics([])
        self.assertEqual(empty["per_item"], [])
        self.assertEqual(empty["prompts"], 0)
        self.assertTrue(math.isnan(empty["weighted_smape"]))
        self.assertTrue(math.isnan(empty["weighted_nmae"]))

    def test_sparse_item_report(self):
        # The space in the synthetic completion "0. 1 mmhg" decodes to 0.1.
        zero, point_one, ten, twenty = "0 mmhg", "0. 1 mmhg", "10 mmhg", "20 mmhg"
        self.assertEqual(parse_lab_value(point_one, "mmhg"), 0.1)
        point_one_smape = 200 * 9.9 / 10.1  # Truth 10, prediction 0.1.
        cases = [
            # truth, generated text, numeric count, retained count, bounds, MAE, SMAPE
            ([10], [ten], 1, 1, (10, 10), 0, 0),
            ([10], [point_one], 1, 1, (10, 10), 9.9, point_one_smape),
            ([10, 10], [ten, ten], 2, 2, (10, 10), 0, 0),
            ([10, 10], [ten, zero], 2, 2, (10, 10), 5, 100),
            ([10, 10], [zero, point_one], 2, 2, (10, 10), 9.95, (200 + point_one_smape) / 2),
            # BUG: The following scenario shows that prefect predictions are liable 
            # to be dropped from the 1/99th percentile bounds retention rule. 
            # see logic @pyhealth/metrics/labtop.py line 45
            ([10, 20], [ten, twenty], 2, 0, (10.1, 19.9), math.nan, math.nan),
            ([10, 20], [zero, point_one], 2, 0, (10.1, 19.9), math.nan, math.nan),
            ([10], ["0 bpm"], 0, 0, (math.nan, math.nan), math.nan, math.nan),
            ([10, 20], [ten, "0 bpm"], 1, 1, (10, 10), 0, 0),
        ]
        for truths, texts, numeric, retained, bounds, mae, smape in cases:
            with self.subTest(truths=truths, generated=texts):
                records = [("pressure", truth, parse_lab_value(text, "mmhg"))
                           for truth, text in zip(truths, texts, strict=True)]
                result = labtop_metrics(records)
                row, = result["per_item"]
                self.assertEqual(row["item"], "pressure")
                self.assertEqual(result["prompts"], len(truths))
                self.assertEqual(result["numeric_predictions"], numeric)
                self.assertEqual(result["retained"], retained)
                self.assertEqual(row["count"], retained)
                for actual, expected in zip(
                    (row["min_v"], row["max_v"], row["mae"], row["smape"],
                     result["weighted_smape"], row["nmae"], result["weighted_nmae"]),
                    (*bounds, mae, smape, smape, math.nan, math.nan), strict=True
                ):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertAlmostEqual(actual, expected)
