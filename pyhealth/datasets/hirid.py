# name: Badea Alamidi
# NetID: alamidi2
# Paper title: "LabTOP: A Unified Model for Lab Test Outcome Prediction on
# Electronic Health Records Paper"
# Paper link: https://proceedings.mlr.press/v287/im25a.html
# Description of dataset: HiRID is a freely accessible critical care dataset
# containing data relating to almost 34 thousand admissions to the Department
# of Intensive Care Medicine of the Bern University Hospital, Switzerland.
"""HiRID source data exposed through the standard PyHealth dataset API.

For the LabTOP-compatible layout, extract or arrange the published HiRID files
under one ``root`` as follows::

    root/
    |-- general_table.csv
    |-- hirid_variable_reference.csv
    |-- observation_tables/parquet/*.parquet
    `-- pharma_records/parquet/*.parquet

The two CSV files come from ``reference_data.tar.gz``; ``general_table.csv``
is required here, while ``hirid_variable_reference.csv`` is needed by the
downstream LabTOP verbalization task. Prefer the Parquet archives
``raw_stage/observation_tables_parquet.tar.gz`` and
``raw_stage/pharma_records_parquet.tar.gz`` for the event tables. The raw-stage
CSV archives and the merged/imputed-stage archives are not used by this
adapter. HiRID's ``patientid`` identifies an ICU admission, not a linkable
person across admissions, so it is also the PyHealth patient ID.

This adapter deliberately stops at source normalization. LabTOP-specific
cohort selection, item-frequency filtering, code-to-description mapping,
relative ICU time, and text/token construction belong to the labtop task / processor.
"""

import logging
from pathlib import Path

import narwhals as nw

from .base_dataset import SOURCE_ROW, BaseDataset

logger = logging.getLogger(__name__)

REQUIRED_TABLES = (
    "general_table",
    "observation_tables",
    "pharma_records",
)
VARIABLE_REFERENCE_FILE = "hirid_variable_reference.csv"


class HIRIDDataset(BaseDataset):
    """Loads the raw HiRID 1.1.1 tables as PyHealth events.
    HiRID contains data related to almost 34 thousand patient admissions to the
    Department of Intensive Care Medicine of the 
    Bern University Hospital, Switzerland (ICU)

    Each raw row becomes an event: observations use ``datetime`` and medication
    records use ``givenat``. Records are deduped and uniquely identified by
    ``(patientid, timestamp, item ID)``.

    LabTOP uses all three tables, so each name in :data:`REQUIRED_TABLES` must
    be passed. ``general_table`` contributes one admission event per patient.
    ``hirid_variable_reference.csv`` is not converted to patient events. Its
    expected location is exposed through :attr:`variable_reference_path`, but
    only the downstream LabTOP task that uses its code descriptions should
    require it.

    Args:
        root: Directory containing the extracted HiRID files in the layout
            shown in the module documentation.
        tables: Tables to load. This argument is required and must contain
            ``general_table``, ``observation_tables``, and ``pharma_records``.
        dataset_name: Dataset name used in cache identity and logging.
        config_path: Optional alternative dataset YAML configuration.
        **kwargs: Forwarded to :class:`BaseDataset`, including ``cache_dir``,
            ``num_workers``, and ``dev``.

    Examples:
        >>> from pyhealth.datasets import HIRIDDataset
        >>> dataset = HIRIDDataset(
        ...     root="/path/to/hirid/1.1.1",
        ...     tables=[
        ...         "general_table",
        ...         "observation_tables",
        ...         "pharma_records",
        ...     ],
        ... )  # doctest: +SKIP
        >>> patient = dataset.get_patient(  # doctest: +SKIP
        ...     dataset.unique_patient_ids[0]
        ... )
        >>> observations = patient.get_events("observation_tables")  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str | Path,
        tables: list[str],
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        selected_tables = list(dict.fromkeys(tables))
        if not selected_tables:
            raise ValueError("tables must contain at least one HiRID table")
        unsupported = [
            table for table in selected_tables if table not in REQUIRED_TABLES
        ]
        if unsupported:
            raise ValueError(
                "Unsupported HiRID table(s): "
                f"{', '.join(unsupported)}. Expected only {REQUIRED_TABLES}."
            )
        missing = [table for table in REQUIRED_TABLES if table not in selected_tables]
        if missing:
            raise ValueError(
                "LabTOP-compatible HiRID loading requires every table in "
                f"{REQUIRED_TABLES}; missing: {', '.join(missing)}"
            )

        # A canonical order keeps event ordering and dataset identity stable
        # even when callers list the required tables in a different order.
        selected_tables = list(REQUIRED_TABLES)
        root_path = Path(root)
        self.variable_reference_path = root_path / VARIABLE_REFERENCE_FILE

        if config_path is None:
            logger.info("No config path provided, using default HiRID config")
            config_path = Path(__file__).parent / "configs" / "hirid.yaml"

        super().__init__(
            root=str(root_path),
            tables=selected_tables,
            dataset_name=dataset_name or "hirid",
            config_path=str(config_path),
            **kwargs,
        )

    @staticmethod
    def _deduplicate_events(
        frame: nw.LazyFrame, *, timestamp: str, item_id: str
    ) -> nw.LazyFrame:
        """Drops untimed rows and keeps the first stay/time/item occurrence."""
        return frame.filter(~nw.col(timestamp).is_null()).unique(
            subset=["patientid", timestamp, item_id],
            keep="first",
            order_by=[SOURCE_ROW],
        )

    def preprocess_observation_tables(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Deduplicates observations by stay, observation time, and raw code."""
        return self._deduplicate_events(
            frame, timestamp="datetime", item_id="variableid"
        )

    def preprocess_pharma_records(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Deduplicates medication rows by stay, administration time, and code."""
        return self._deduplicate_events(
            frame, timestamp="givenat", item_id="pharmaid"
        )
