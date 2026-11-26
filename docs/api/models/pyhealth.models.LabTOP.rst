pyhealth.models.LabTOP
=====================

LabTOP predicts numeric laboratory results from a chronological ICU event
sequence. The example trains on HiRID ICU admissions
and evaluates the best validation epoch on held-out prompts.

.. autoclass:: pyhealth.models.LabTOP
    :members:

.. autoclass:: pyhealth.tasks.labtop.LabTOPHiRIDProcessor
    :members:
    :inherited-members:

.. autofunction:: pyhealth.metrics.labtop.labtop_metrics

Running the HiRID example
-------------------------

Use the HiRID 1.1.1 raw-stage layout documented by
:class:`pyhealth.datasets.HIRIDDataset`. The root contains general_table.csv,
hirid_variable_reference.csv, observation_tables/parquet/ and pharma_records/parquet/.
A small training configuration uses::

    python -m examples.labtop.train /path/to/hirid --lab-only \
        --epochs 1 --max-seq-len 128 --hidden-dim 32 --layers 1 --heads 2 --batch-size 4

The tokenizer defaults to Bio_ClinicalBERT. Pass --tokenizer with a local copy for offline use.

Each HiRID patient ID identifies one admission. Length of stay is measured from
admission to the last timestamp across both event tables. Numeric observations,
including vital signs, are eligible targets. Medication names and dose units
provide context. Names and observation units come from the source-specific
variable reference. Time prefixes contain ICU day, weekday, hour and
ten-minute bin.

Training and evaluation
-----------------------

Patients are split 80/10/10 with seed 42. The processor selects frequent items
on training patients and encodes eligible stays of at least six hours. Each
training batch contains token IDs, lab masks and sequential position IDs, all
right-padded to the longest sequence. The example enables ``--lab-only`` by default for both
training and validation. The event time, table and complete item name form the
header. Loss starts at the first value token and includes the unit, when
present, and the end-of-event marker. Use ``--no-lab-only`` to train on all
non-padding next tokens.

Test prompts contain complete earlier events followed by the time, table and
item of the requested lab. The processor excludes all other events in that
ten-minute bin and reserves generation space independently of the target value.
Evaluation uses the best validation epoch and reports the number of generated
values that parse as numbers alongside the percentile-trimmed errors.
