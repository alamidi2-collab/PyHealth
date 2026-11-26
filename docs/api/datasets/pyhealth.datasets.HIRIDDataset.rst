pyhealth.datasets.HIRIDDataset
=============================

Overview and source
-------------------

``HIRIDDataset`` loads the raw tables of
`HiRID version 1.1.1 <https://physionet.org/content/hirid/1.1.1/>`_
through :class:`~pyhealth.datasets.BaseDataset`. The original resource contains
ICU observations, medications and admission information from Bern University
Hospital. Obtain the data through its official access process before using
this loader; the loader does not download or redistribute clinical records.

This contribution supports the source layout needed by
`LabTOP <https://proceedings.mlr.press/v287/im25a.html>`_. Its preprocessing stops
at event normalization. Cohort selection, variable descriptions, item-frequency
selection, tokenization and prediction prompts are implemented separately by
:class:`~pyhealth.tasks.labtop.LabTOPHiRIDProcessor` and
:class:`~pyhealth.tasks.labtop.LabTOPTask`.

Expected local files
--------------------

The dataset root must contain these extracted files:

* ``general_table.csv``
* ``observation_tables/parquet/`` containing the raw observation Parquet files
* ``pharma_records/parquet/`` containing the raw medication Parquet files

For LabTOP, also place ``hirid_variable_reference.csv`` in the root. The
reference table is used for variable descriptions and units by the processor;
loading raw patient events does not require it. Its location is exposed as
``dataset.variable_reference_path``.

Use the raw-stage Parquet archives. The merged and imputed stages and raw-stage
CSV event archives are not supported by this adapter. The published
``patientid`` identifies an ICU admission rather than a person that can be
linked across admissions; it becomes the PyHealth patient identifier.

Loading and event semantics
---------------------------

.. code-block:: python

    from pyhealth.datasets import HIRIDDataset
    from pyhealth.datasets.hirid import REQUIRED_TABLES

    dataset = HIRIDDataset(
        root="/path/to/hirid/1.1.1",
        tables=list(REQUIRED_TABLES),
        cache_dir="/path/to/local/project-cache",
        num_workers=1,
    )
    patient = dataset.get_patient(dataset.unique_patient_ids[0])
    observations = patient.get_events("observation_tables")
    medication_events = patient.get_events("pharma_records")

All three names in ``REQUIRED_TABLES`` must be provided; missing or unsupported
tables raise ``ValueError``. The loader canonicalizes their order. Its YAML
configuration is ``pyhealth/datasets/configs/hirid.yaml``.

``general_table.admissiontime`` becomes the admission event's ``timestamp``.
Observation ``datetime`` and medication ``givenat`` become event timestamps.
Rows without those event timestamps are removed. Duplicate observations with
the same admission, timestamp and raw variable ID retain the first source row;
medications use the analogous raw medication ID. Observation and medication ID
namespaces are kept separate.

LabTOP usage
------------

.. code-block:: python

    from pyhealth.tasks import LabTOPHiRIDProcessor, LabTOPTask

    processor = LabTOPHiRIDProcessor(
        tokenizer,
        dataset.variable_reference_path,
        max_seq_len=4096,
    )
    processor.fit(dataset.get_patient(pid) for pid in train_ids)
    task = LabTOPTask(processor, lab_only=True)
    training_rows = [
        row for pid in train_ids for row in task(dataset.get_patient(pid))
    ]
    test_prompt_dataset = [
        row
        for pid in test_ids
        for row in processor.make_inference_dataset(
            dataset.get_patient(pid), max_new_tokens=50
        )
    ]

Choose disjoint training, validation and test identifiers before fitting item
selection. HiRID's admission identifiers cannot establish person-level
disjointness across repeated admissions. The processor applies the six-hour
stay requirement, ten-minute time bins and weekday tokens; these are task
operations and do not alter the raw dataset cache.

Testing
-------

``tests/core/test_hirid.py`` uses two entirely invented admissions, CSV reference
information and tiny Parquet event tables in temporary directories. It covers
loading, timestamp mapping, deterministic deduplication, table validation and
LabTOP prompt construction. No real or demo clinical dataset is required.

API
---

.. autoclass:: pyhealth.datasets.HIRIDDataset
    :members:
    :show-inheritance:
