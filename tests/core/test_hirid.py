from datetime import datetime

import pandas as pd

from pyhealth.datasets import HIRIDDataset


def _write_hirid_fixture(root):
    pd.DataFrame(
        {
            "ID": [200, 426],
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
                    "2150-01-01 08:10:00",
                    None,
                    "2150-02-01 09:05:00",
                ]
            ),
            "entertime": pd.to_datetime(
                [
                    "2150-01-01 08:06:00",
                    "2150-01-01 08:07:00",
                    "2150-01-01 08:11:00",
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
            "pharmaid": [426, 426],
            "givenat": pd.to_datetime(
                ["2150-01-01 08:15:00", "2150-01-01 08:15:00"]
            ),
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


def test_hirid_loads_patient_and_deduplicates_events(tmp_path):
    _write_hirid_fixture(tmp_path)
    dataset = HIRIDDataset(
        root=tmp_path,
        tables=["general_table", "observation_tables", "pharma_records"],
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )

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

    assert len(medications) == 1
    assert medications[0].givendose == 1.5
    assert medications[0].doseunit == "mg"


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


def test_hirid_labtop_stay_and_prompt(tmp_path):
    from transformers import BertTokenizerFast

    from pyhealth.tasks.labtop import LabTOPHiRIDProcessor, collate_labtop_prompts

    _write_hirid_fixture(tmp_path)
    # An ID may identify both a drug and an observation. A dose recorded
    # between two observations must be in the later observation's history.
    ref_path = tmp_path / "hirid_variable_reference.csv"
    reference = pd.read_csv(ref_path)
    reference.loc[1, "ID"] = 200
    reference.to_csv(ref_path, index=False)
    drug_path = tmp_path / "pharma_records/parquet/part-0.parquet"
    drugs = pd.read_parquet(drug_path)
    drugs["pharmaid"] = 200
    drugs.to_parquet(drug_path, index=False)
    obs_path = tmp_path / "observation_tables/parquet/part-0.parquet"
    obs = pd.read_parquet(obs_path)
    obs.loc[2, "datetime"] = pd.Timestamp("2150-01-02 00:10:00")
    obs.to_parquet(obs_path, index=False)

    dataset = HIRIDDataset(tmp_path, ["general_table", "observation_tables", "pharma_records"],
                           cache_dir=tmp_path / "cache")
    (tmp_path / "vocab.txt").write_text("\n".join(
        "[PAD] [UNK] [CLS] [SEP] [MASK] heart rate dobutamine mg / min".split()
    ))
    tokenizer = BertTokenizerFast.from_pretrained(tmp_path)
    processor = LabTOPHiRIDProcessor(tokenizer, dataset.variable_reference_path, max_seq_len=128)
    patient = dataset.get_patient("1")
    processor.fit([patient])
    assert processor.preprocess(dataset.get_patient("2")) == []  # five-minute stay
    row, = processor.preprocess(patient, lab_only=True)
    prompt, = processor.make_inference_dataset(patient)
    assert prompt["item"] == "heart rate"
    assert prompt["unit"] == "/min"
    assert prompt["lab_value"] == 81
    assert prompt["mean_previous_values"] == 80
    assert prompt["prompt_tokens"] + prompt["lab_tokens"] == row["tokens"]
    prefix, _, _, _ = collate_labtop_prompts([prompt])
    tokens = tokenizer.convert_ids_to_tokens(prefix[0].tolist())
    assert tokens[-2:] == ["heart", "rate"]
    drug_start = tokens.index("dobutamine")
    assert tokens[drug_start:drug_start + 3] == ["dobutamine", "mg", "|endofevent|"]
    assert row["types"][drug_start] == 0
    assert row["types"][tokens.index("heart")] == 1
    stamp = ["[DAY_2]", "[FRI]", "[00h]", "[10m]"]
    assert tokenizer.convert_ids_to_tokens(prompt["lab_tokens"][:4]) == stamp
