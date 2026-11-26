"""Two invented HiRID admissions through the real parser and LabTOP processor.

The shared fixture builds the actual event cache once. Pytest reports that
Dask initialization as fixture setup time, separately from test calls; it is
not removed or replaced with a parser mock.
"""

from datetime import datetime

import pandas as pd
import pytest
from transformers import BertTokenizerFast

from pyhealth.datasets import HIRIDDataset
from pyhealth.tasks.labtop import LabTOPHiRIDProcessor, collate_labtop_prompts


def _write_hirid_fixture(root):
    pd.DataFrame(
        {
            # Observation and medication namespaces may reuse a raw code.
            "ID": [200, 200],
            "Variable Name": ["heart rate", "dobutamine"],
            "Unit": ["/min", "mg"],
            "Source Table": ["Observation", "Pharma"],
        }
    ).to_csv(root / "hirid_variable_reference.csv", index=False)

    pd.DataFrame(
        {
            "patientid": [1, 2],
            "admissiontime": ["2150-01-01 08:00:00", "2150-02-01 09:00:00"],
            "sex": ["F", "M"],
            "age": [65, 70],
            "discharge_status": ["alive", "dead"],
        }
    ).to_csv(root / "general_table.csv", index=False)

    observation_dir = root / "observation_tables" / "parquet"
    observation_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "patientid": [1, 1, 1, 1, 2],
            "datetime": pd.to_datetime(
                [
                    "2150-01-01 08:05:00",
                    "2150-01-01 08:05:00",
                    "2150-01-02 00:10:00",
                    None,
                    "2150-02-01 09:05:00",
                ]
            ),
            "entertime": pd.to_datetime(
                [
                    "2150-01-01 08:06:00",
                    "2150-01-01 08:07:00",
                    "2150-01-02 00:11:00",
                    "2150-01-01 08:12:00",
                    "2150-02-01 09:06:00",
                ]
            ),
            "status": [8, 8, 8, 8, 8],
            "stringvalue": [None, None, None, None, None],
            "type": [None, None, None, None, None],
            "value": [80.0, 999.0, 81.0, 500.0, 70.0],
            "variableid": [200, 200, 200, 201, 200],
        }
    ).to_parquet(observation_dir / "part-0.parquet", index=False)

    pharma_dir = root / "pharma_records" / "parquet"
    pharma_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "patientid": [1, 1],
            "pharmaid": [200, 200],
            "givenat": pd.to_datetime(["2150-01-01 08:15:00", "2150-01-01 08:15:00"]),
            "enteredentryat": pd.to_datetime(
                ["2150-01-01 08:16:00", "2150-01-01 08:17:00"]
            ),
            "givendose": [1.5, 9.5],
            "cumulativedose": [1.5, 9.5],
            "fluidamount_calc": [0.0, 0.0],
            "cumulfluidamount_calc": [0.0, 0.0],
            "doseunit": ["mg", "mg"],
            "route": ["iv", "iv"],
            "infusionid": [100, 101],
            "typeid": [1, 1],
            "subtypeid": [8.0, 8.0],
            "recordstatus": [8, 8],
        }
    ).to_parquet(pharma_dir / "part-0.parquet", index=False)


@pytest.fixture(scope="module")
def loaded_hirid(tmp_path_factory):
    """Parse two invented admissions once, with normal Dask and Parquet I/O."""
    root = tmp_path_factory.mktemp("hirid_synthetic")
    _write_hirid_fixture(root)
    dataset = HIRIDDataset(
        root=root,
        tables=["general_table", "observation_tables", "pharma_records"],
        cache_dir=root / "cache",
        num_workers=1,
    )
    # Warm the real event cache during module-fixture setup. Tests below only
    # read it and do not mutate the source files or bypass any loading methods.
    dataset.get_patient("1")
    return dataset


def test_hirid_loads_patient_and_deduplicates_events(loaded_hirid):
    dataset = loaded_hirid
    patient = dataset.get_patient("1")

    observations = patient.get_events("observation_tables")
    medications = patient.get_events("pharma_records")
    admissions = patient.get_events("general_table")

    assert len(admissions) == 1
    assert admissions[0].timestamp == datetime(2150, 1, 1, 8, 0)
    assert admissions[0].sex == "F"
    assert admissions[0].age == "65"

    assert len(observations) == 2
    assert [event.value for event in observations] == [80.0, 81.0]
    assert observations[0].variableid == 200
    assert [event.timestamp for event in observations] == [
        datetime(2150, 1, 1, 8, 5),
        datetime(2150, 1, 2, 0, 10),
    ]

    assert len(medications) == 1
    assert medications[0].givendose == 1.5
    assert medications[0].doseunit == "mg"
    assert medications[0].pharmaid == 200
    assert medications[0].timestamp == datetime(2150, 1, 1, 8, 15)


def test_hirid_canonicalizes_required_table_order(tmp_path):
    _write_hirid_fixture(tmp_path)
    # Code descriptions belong to the task, so Patient construction does not
    # require the reference lookup file.
    (tmp_path / "hirid_variable_reference.csv").unlink()
    dataset = HIRIDDataset(
        root=tmp_path,
        tables=["pharma_records", "observation_tables", "general_table"],
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )

    assert dataset.tables == [
        "general_table",
        "observation_tables",
        "pharma_records",
    ]
    assert dataset.variable_reference_path == (
        tmp_path / "hirid_variable_reference.csv"
    )


def test_hirid_rejects_missing_required_table(tmp_path):
    try:
        HIRIDDataset(
            root=tmp_path,
            tables=["general_table", "observation_tables"],
        )
    except ValueError as error:
        assert "missing: pharma_records" in str(error)
    else:
        raise AssertionError("A missing LabTOP source table should not be accepted")


def test_hirid_rejects_non_event_table(tmp_path):
    try:
        HIRIDDataset(
            root=tmp_path,
            tables=[
                "general_table",
                "observation_tables",
                "pharma_records",
                "merged_stage",
            ],
        )
    except ValueError as error:
        assert "Unsupported HiRID table" in str(error)
    else:
        raise AssertionError("Non-LabTOP tables should not be accepted")


def test_hirid_labtop_stay_and_prompt(loaded_hirid, tmp_path):
    # The shared data already has a >=6-hour stay, a five-minute stay, and
    # a medication code also used by observations. No fixture mutation needed.
    dataset = loaded_hirid
    (tmp_path / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nheart\nrate\ndobutamine\nmg\n/\nmin"
    )
    tokenizer = BertTokenizerFast.from_pretrained(tmp_path, local_files_only=True)
    processor = LabTOPHiRIDProcessor(
        tokenizer, dataset.variable_reference_path, max_seq_len=128
    )
    patient = dataset.get_patient("1")
    processor.fit([patient])
    assert processor.preprocess(dataset.get_patient("2")) == []  # five-minute stay
    (row,) = processor.preprocess(patient, lab_only=True)
    (prompt,) = processor.make_inference_dataset(patient)
    assert prompt["item"] == "heart rate"
    assert prompt["unit"] == "/min"
    assert prompt["lab_value"] == 81
    assert prompt["mean_previous_values"] == 80
    assert prompt["prompt_tokens"] + prompt["lab_tokens"] == row["tokens"]
    prefix, _, _, _ = collate_labtop_prompts([prompt])
    tokens = tokenizer.convert_ids_to_tokens(prefix[0].tolist())
    assert tokens[-2:] == ["heart", "rate"]
    assert "dobutamine" in tokens
    assert row["types"][tokens.index("dobutamine")] == 0
    assert row["types"][tokens.index("heart")] == 1
    stamp = ["[DAY_2]", "[FRI]", "[00h]", "[10m]"]
    assert tokenizer.convert_ids_to_tokens(prompt["lab_tokens"][:4]) == stamp
