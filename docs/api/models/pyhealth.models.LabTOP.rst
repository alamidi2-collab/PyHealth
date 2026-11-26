pyhealth.models.LabTOP
=====================

LabTOP predicts numeric laboratory values from chronological ICU event text.
This adapter inherits :class:`pyhealth.models.BaseModel` and retains the
existing tokenizer-first constructor. It supports a native PyHealth sample
dataset, shifted autoregressive loss, an embedding-based forward method, and
greedy value generation with restored model modes.

Original paper: `Im, Oh, and Choi, CHIL 2025
<https://proceedings.mlr.press/v287/im25a.html>`_.

.. autoclass:: pyhealth.models.LabTOP
    :members:

Offline full pipeline
---------------------

After installing the checkout, run from the repository root::

    python examples/eicu_lab_prediction_labtop.py --synthetic --epochs 2 \
        --max-new-tokens 8 --history-keep 1 .75 .5 --output /tmp/labtop_smoke

Invented eICU-shaped records and a local tokenizer exercise dataset loading,
:class:`pyhealth.tasks.labtop.LabTOPTask`, a native sample dataset, model
training, validation checkpoint selection, and held-out prompt evaluation.
Synthetic scores demonstrate execution only.

``examples/labtop/README.md`` documents longer runs, recorded synthetic results,
clinical data commands, checkpoint compatibility, and all reproduction limits.
``examples/my_replication.py`` is an equivalent full-pipeline entry point.

Checkpoint evaluation
---------------------

Saved runs contain ``best_model.pt`` and the matching ``tokenizer/`` directory.
Evaluate without training::

    python examples/eicu_lab_prediction_labtop.py --synthetic --eval-only \
        --checkpoint /tmp/labtop_smoke/best_model.pt --max-new-tokens 8 \
        --history-keep 1 .75 .5 --output /tmp/labtop_reload

For local clinical data, replace ``--synthetic`` with the eICU root. Use
``--dataset hirid`` for the raw HiRID layout documented in
:class:`pyhealth.datasets.HIRIDDataset`. Keep the checkpoint's original
architecture, tokenizer and patient split. Legacy adapter state dictionaries
may omit only the newly added BaseModel device marker; incompatible weights
raise an error.

Held-out inference and metrics
------------------------------

Training patients alone determine frequent-item selection. Validation token
loss selects the checkpoint. The ``test_prompt_dataset`` contains only test
patients. Prompt construction retains complete earlier events, excludes the
entire target time bin, and reserves the requested generation budget without
using the answer's length. The target item name stays in the input while its
value, unit and delimiter stay in the labels.

.. autofunction:: pyhealth.metrics.labtop.parse_lab_value

.. autofunction:: pyhealth.metrics.labtop.labtop_metrics

Errors are conditional on numeric predictions. Always inspect parse coverage,
retained counts and undefined scores. Last-value and historical-mean baselines
are evaluated separately and omit targets with missing same-unit history.
Saved CSV/JSON outputs contain aggregates rather than patient-level records.

Robustness experiment and scope
-------------------------------

``--history-keep 1 .75 .5`` removes complete previous events while holding the
checkpoint, test targets, target headers and generation settings fixed. Matched
random draws create nested retained sets. This is an inference robustness
experiment; lower errors on fewer parseable targets do not imply improvement.

The current lab-only mask includes lab item names, values, units and delimiters,
matching the released source mask. The default objective is full autoregressive
loss, while ``--lab-only`` uses that mask. The port does not reconstruct every
paper demographic or digit-tokenization choice. Its HiRID processor targets
numeric observations including vitals, which differs from the final paper's
selected laboratory cohort. Clinical reproduction requires matching those
choices and evaluating the actual trained clinical checkpoint.
