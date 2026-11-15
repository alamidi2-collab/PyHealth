import pandas as pd

from pyhealth.datasets import HIRIDDataset


def _write_hirid_fixture(root):
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
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )

    patient = dataset.get_patient("1")

    observations = patient.get_events("observation_tables")
    medications = patient.get_events("pharma_records")

    assert len(observations) == 2
    assert [event.value for event in observations] == [80.0, 81.0]
    assert observations[0].variableid == 200

    assert len(medications) == 1
    assert medications[0].givendose == 1.5
    assert medications[0].doseunit == "mg"


def test_hirid_explicit_table_selection(tmp_path):
    _write_hirid_fixture(tmp_path)
    dataset = HIRIDDataset(
        root=tmp_path,
        tables=["observation_tables"],
        cache_dir=tmp_path / "cache",
        num_workers=1,
    )

    assert dataset.tables == ["observation_tables"]
    patient = dataset.get_patient("1")
    assert len(patient.get_events("observation_tables")) == 2


def test_hirid_rejects_non_event_table(tmp_path):
    try:
        HIRIDDataset(root=tmp_path, tables=["general_table"])
    except ValueError as error:
        assert "Unsupported HiRID table" in str(error)
    else:
        raise AssertionError("general_table should not be a selectable HiRID table")
