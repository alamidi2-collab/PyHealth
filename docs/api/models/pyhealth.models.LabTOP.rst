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

From the PyHealth repository root, activate your Python environment. For a
fresh checkout, install the repository into that environment once::

    python -m pip install -e .

HiRID 1.1.1 is from PhysioNet.com. In Bash, set ``HIRID_DOWNLOAD`` to the
directory containing ``reference_data.tar.gz`` and ``raw_stage/``. On WSL 
(or whatever is equivalent for your shell). The following commands put the 
extracted data, reusable cache and logs alongside the download::

    HIRID_DOWNLOAD="/path/to/hirid-1.1.1"
    RUN_DIR="$HIRID_DOWNLOAD/labtop-run"
    HIRID_ROOT="$RUN_DIR/hirid"
    mkdir -p "$HIRID_ROOT"
    LOG="$RUN_DIR/labtop-$(date +%Y%m%d-%H%M%S).log"
    echo "Run log: $LOG"
    set -o pipefail

Extract the reference tables and the two raw parquet archives once. If you
already have the extracted layout expected by :class:`pyhealth.datasets.HIRIDDataset`,
set ``HIRID_ROOT`` to that directory and skip this block::

    (
        set -ex
        tar -xzf "$HIRID_DOWNLOAD/reference_data.tar.gz" -C "$HIRID_ROOT" general_table.csv hirid_variable_reference.csv
        tar -xzf "$HIRID_DOWNLOAD/raw_stage/observation_tables_parquet.tar.gz" -C "$HIRID_ROOT"
        tar -xzf "$HIRID_DOWNLOAD/raw_stage/pharma_records_parquet.tar.gz" -C "$HIRID_ROOT"
    ) 2>&1 | tee -a "$LOG"

After extraction succeeds, run one epoch and evaluate it on the held-out
patients. Run this in the same shell from the repository root::

    python -u -m examples.labtop.train "$HIRID_ROOT" --lab-only --epochs 1 \
        --cache-dir "$RUN_DIR/cache" 2>&1 | tee -a "$LOG"

The example loads the parquet data, creates the patient splits (80/10/10 with seed 42),
fits the itemfilter on training patients, and builds training and validation sequences.
After training, it uses the best validation epoch to generate predictions
for the test prompts. The terminal and log both show the stay counts, validation
loss and final metrics, including the number of prompts and numeric predictions.
The event cache persists across runs. The example keeps model weights in memory
through evaluation. For the 15-epoch configuration, use ``--epochs 15``.
The tokenizer defaults to Bio_ClinicalBERT. For offline use, supply
``--tokenizer /path/to/Bio_ClinicalBERT``.

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
training and validation. This makes event time, table and complete item name form the
header of an event. Then, Prediction loss starts at the first value token and includes the unit, when
present, and the end-of-event marker. Use ``--no-lab-only`` to train on all
non-padding next tokens.

Test prompts contain complete earlier events followed by the time, table and
item of the requested lab. The processor excludes all other events in that
ten-minute bin and reserves generation space independently of the target value.
Evaluation uses the best validation epoch and reports the number of generated
values that parse as numbers alongside the percentile-trimmed errors.
