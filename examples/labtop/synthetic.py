"""Invented eICU-shaped records and an offline tokenizer for LabTOP examples.

This module never reads a clinical dataset or downloads a tokenizer. The values
are a simple counting pattern in invented units, useful only for exercising data
loading, training, ablations and evaluation. They cannot support a clinical or
paper-reproduction claim.
"""

from __future__ import annotations

import csv
import json
import random
from numbers import Integral
from pathlib import Path
from typing import Any

import yaml
from transformers import BertTokenizerFast

_TABLE_NAMES = ("patient", "lab", "medication", "treatment")


def build_tokenizer(root: Path) -> Path:
    """Write a tiny local BERT tokenizer and return its directory.

    Args:
        root: Directory in which to create tokenizer files.

    Returns:
        The directory accepted by ``AutoTokenizer.from_pretrained`` with
        ``local_files_only=True``. Padding has ID zero.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "lab",
        "glucose",
        "sodium",
        "units",
        "medication",
        "treatment",
        "synthetic",
        "saline",
        "monitoring",
        ".",
        "-",
        "/",
        *[str(number) for number in range(10)],
        *[f"##{number}" for number in range(10)],
    ]
    (root / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast.from_pretrained(root, local_files_only=True)
    if tokenizer.pad_token_id != 0:
        raise ValueError("The synthetic tokenizer must use padding ID zero")
    tokenizer.save_pretrained(root)
    return root


def _write_table(
    root: Path, name: str, columns: list[str], rows: list[dict[str, Any]]
) -> None:
    """Write complete configured CSV columns with blank unused attributes."""
    with (root / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_synthetic_eicu(root: Path, n_patients: int = 20, seed: int = 42) -> Path:
    """Create deterministic invented eICU CSVs and an offline tokenizer.

    Each synthetic patient has one twelve-hour stay, two context events and
    twelve lab observations (six each for glucose and sodium). For each item,
    values cycle through 1, 2, 3 and 4 with a seeded patient-specific phase.
    ``units`` is intentionally an invented unit; these are not plausible
    physiological measurements. Context precedes the first target, and targets
    occupy distinct ten-minute bins.

    Args:
        root: New or empty directory receiving the CSVs and ``tokenizer/``.
        n_patients: Positive number of invented patients. Use at most five in
            unit tests. Training examples generally need at least three for
            disjoint training, validation and test splits.
        seed: Integer seed controlling the deterministic counting phases.

    Returns:
        Path to the offline tokenizer directory under ``root``.

    Raises:
        ValueError: If the count is invalid.
        TypeError: If the seed is not an integer.
        FileExistsError: If any destination CSV exists, to avoid overwriting
            an existing dataset.
    """
    if (
        isinstance(n_patients, bool)
        or not isinstance(n_patients, Integral)
        or n_patients < 1
    ):
        raise ValueError("n_patients must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / f"{name}.csv").exists() for name in _TABLE_NAMES):
        raise FileExistsError("Synthetic generation requires absent destination CSVs")

    # Read the installed checkout's schema so unused required attributes are
    # present too. No values are read from a real or demo patient table.
    config_path = (
        Path(__file__).resolve().parents[2]
        / "pyhealth"
        / "datasets"
        / "configs"
        / "eicu.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["tables"]
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in _TABLE_NAMES}
    rng = random.Random(int(seed))
    for index in range(int(n_patients)):
        stay = f"SYNTHETIC_STAY_{index + 1:04d}"
        tables["patient"].append(
            {
                "uniquepid": f"SYNTHETIC_PATIENT_{index + 1:04d}",
                "patientunitstayid": stay,
                "patienthealthsystemstayid": f"SYNTHETIC_ADMISSION_{index + 1:04d}",
                "gender": "Synthetic",
                "age": 40 + 5 * (index % 8),
                "ethnicity": "Synthetic",
                "hospitalid": "SYNTHETIC_HOSPITAL",
                "wardid": "SYNTHETIC_WARD",
                "apacheadmissiondx": "synthetic example",
                "hospitaladmittime24": "08:00:00",
                "hospitaladmitoffset": 0,
                "hospitaldischargeyear": 2150,
                "hospitaldischargeoffset": 720,
                "hospitaldischargestatus": "Synthetic",
                "hospitaldischargetime24": "20:00:00",
                "unitadmittime24": "08:00:00",
                "unitvisitnumber": 1,
                "unitstaytype": "synthetic",
                "unitdischargeoffset": 720,
                "unitdischargestatus": "Synthetic",
                "unitdischargetime24": "20:00:00",
            }
        )
        tables["medication"].append(
            {
                "patientunitstayid": stay,
                "drugstartoffset": 0,
                "drugstopoffset": 720,
                "drugname": "synthetic saline",
                "dosage": 1,
                "routeadmin": "synthetic",
                "frequency": "synthetic",
                "prn": False,
                "drugordercancelled": False,
            }
        )
        tables["treatment"].append(
            {
                "patientunitstayid": stay,
                "treatmentoffset": 10,
                "treatmentstring": "synthetic monitoring",
                "activeupondischarge": False,
            }
        )
        phase = rng.randrange(4)
        for measurement in range(12):
            item_index = measurement % 2
            value = (measurement // 2 + phase + item_index) % 4 + 1
            tables["lab"].append(
                {
                    "patientunitstayid": stay,
                    "labresultoffset": 20 + 20 * measurement,
                    "labname": ("glucose", "sodium")[item_index],
                    "labresult": value,
                    "labresulttext": str(value),
                    "labmeasurenamesystem": "units",
                    "labmeasurenameinterface": "units",
                    "labtypeid": 1,
                }
            )

    for name, rows in tables.items():
        columns = list(config[name]["attributes"])
        if name == "patient":
            columns.insert(0, config[name]["patient_id"])
        _write_table(root, name, columns, rows)
    metadata = {
        "data_source": "synthetic",
        "purpose": "Execution check only; not clinical or reproduction evidence",
        "generator": "examples.labtop.synthetic.write_synthetic_eicu",
        "patients": int(n_patients),
        "seed": int(seed),
        "labs_per_patient": 12,
        "unit": "units (invented)",
        "pattern": "Per-item values cycle 1,2,3,4 with seeded starting phase",
    }
    (root / "SYNTHETIC_DATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return build_tokenizer(root / "tokenizer")
