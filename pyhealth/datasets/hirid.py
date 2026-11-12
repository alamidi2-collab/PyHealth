# name: Badea Alamidi
# NetID: alamidi2
# Paper title: "LabTOP: A Unified Model for Lab Test Outcome Prediction on
# Electronic Health Records Paper"
# Paper link: https://proceedings.mlr.press/v287/im25a.html
# Description of dataset: HiRID is a freely accessible critical care dataset
# containing data relating to almost 34 thousand admissions to the Department
# of Intensive Care Medicine of the Bern University Hospital, Switzerland.
"""HiRID source data exposed through the standard PyHealth dataset API.

HiRID 1.1.1 publishes one row per ICU admission in ``general_table.csv`` and
sharded Parquet event data under ``observation_tables/parquet`` and
``pharma_records/parquet``. HiRID's ``patientid`` identifies an ICU admission,
not a linkable person across admissions, so it is also the PyHealth patient ID.

This adapter deliberately stops at source normalization. LabTOP-specific
cohort selection, item-frequency filtering, code-to-description mapping,
relative ICU time, and text/token construction belong to a task or processor.
"""

import logging
from pathlib import Path

import narwhals as nw

from .base_dataset import SOURCE_ROW, BaseDataset

logger = logging.getLogger(__name__)

DEFAULT_TABLES = ["general_table", "observation_tables", "pharma_records"]


class HIRIDDataset(BaseDataset):
    """Loads the raw HiRID 1.1.1 tables as PyHealth events.

    The expected directory layout is the one produced by unpacking the three
    HiRID archives used by LabTOP::

        root/
            general_table.csv
            hirid_variable_reference.csv
            observation_tables/parquet/*.parquet
            pharma_records/parquet/*.parquet

    Each raw row becomes an event. ``general_table`` is an admission event,
    observations use ``datetime``, and medication records use ``givenat``.
    Event-table duplicates are removed using ``(patientid, timestamp, item
    ID)`` while retaining the first source row, matching the placement of
    LabTOP's deduplication but using raw IDs because description mapping is
    intentionally deferred.

    No minimum-length-of-stay cohort is applied, no discharge time is inferred
    from the final event, and events are not restricted to a derived ICU
    interval. Those choices alter a prediction cohort and belong in a task.

    Args:
        root: Directory containing the unpacked HiRID files.
        tables: Tables to load. Defaults to all three raw clinical tables.
            When supplied, only those tables are loaded; use
            ``"general_table"`` explicitly when admission metadata is needed.
        dataset_name: Dataset name used in cache identity and logging.
        config_path: Optional alternative dataset YAML configuration.
        **kwargs: Forwarded to :class:`BaseDataset`, including ``cache_dir``,
            ``num_workers``, and ``dev``.

    Examples:
        >>> from pyhealth.datasets import HIRIDDataset
        >>> dataset = HIRIDDataset(
        ...     root="/path/to/hirid/1.1.1",
        ...     tables=["general_table", "observation_tables"],
        ... )  # doctest: +SKIP
        >>> patient = dataset.get_patient(  # doctest: +SKIP
        ...     dataset.unique_patient_ids[0]
        ... )
        >>> observations = patient.get_events("observation_tables")  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str | Path,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        selected_tables = list(
            dict.fromkeys(DEFAULT_TABLES if tables is None else tables)
        )
        if not selected_tables:
            raise ValueError("tables must contain at least one HiRID table")

        if config_path is None:
            logger.info("No config path provided, using default HiRID config")
            config_path = Path(__file__).parent / "configs" / "hirid.yaml"

        super().__init__(
            root=str(root),
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
