# LabTOP: held-out inference, evaluation, and a reproducible robustness example

This contribution connects PyHealth ICU events to LabTOP's causal language model,
constructs a **test** prompt dataset, and evaluates generated numerical outcomes.
The source paper is [Im, Oh, and Choi, CHIL 2025](https://proceedings.mlr.press/v287/im25a.html).
The reference implementation was reviewed at
[`fa2e66749af84338f6cef7ba3c5918e09719c0e4`](https://github.com/sujeongim/LabTOP/tree/fa2e66749af84338f6cef7ba3c5918e09719c0e4).

## What this example finishes

- `LabTOP(BaseModel)` retains the group's tokenizer-first interface and transformer
  parameter names, with validation and an embedding-based forward method.
- `LabTOPTask(BaseTask)` wraps a training-fitted eICU or HiRID processor and creates
  native PyHealth sample datasets. An indexed `Subset` view allows deterministic
  shuffled loading of PyHealth's indexable iterable sample dataset.
- `test_prompt_dataset` uses only the **test patient IDs**. Training patients alone
  determine frequent-item selection. Validation token loss selects the checkpoint.
- Each inference prefix contains complete earlier events plus the target's time,
  table and item name. Target values stay outside model inputs; all events in the
  target's ten-minute bin are excluded. The actual generation budget is reserved
  independently of answer length.
- `evaluate()` parses generated values, computes per-item errors, and saves model
  and historical-baseline results separately, with numeric coverage and denominator
  counts. It never substitutes a baseline for model predictions.
- The runnable ablation removes complete historical events at inference while
  fixing checkpoint, target set, target header and generation settings.

The example is a limited port and reproducibility attempt, not a claim that every
paper preprocessing choice or published clinical result has been reproduced.

## Installation and offline smoke run

Use Python 3.12 or 3.13 and install this checkout. PyHealth's complete installation
can be larger than the tiny demonstration itself.

```bash
python -m pip install -e . pytest
python examples/eicu_lab_prediction_labtop.py --synthetic --epochs 2 \
  --max-new-tokens 8 --history-keep 1 .75 .5 --output /tmp/labtop_smoke
```

`--synthetic` creates invented eICU-shaped CSVs and a local BERT tokenizer inside a
cleaned-up temporary directory. It loads those files through the actual
`eICUDataset`, then executes the task, native sample dataset, model training,
validation selection, test prompt creation and evaluation. It downloads neither
clinical data nor tokenizer weights. No synthetic values represent patients or
physiological measurements. Two epochs demonstrate execution; unparseable
predictions and undefined scores are possible for an undertrained language model.

For the recorded toy demonstration:

```bash
python examples/eicu_lab_prediction_labtop.py --synthetic --epochs 40 \
  --hidden-dim 32 --layers 1 --heads 2 --max-seq-len 128 \
  --batch-size 4 --accumulation-steps 1 --learning-rate 0.003 \
  --max-new-tokens 8 --history-keep 1 .75 .5 --output /tmp/labtop_demo
```

The same pipeline is available through `python -m examples.labtop.train` and the
rubric's full-pipeline entry point, `python examples/my_replication.py`.
Choose a fresh output directory; existing results are protected from overwriting.

## Checkpoint evaluation without retraining

```bash
python examples/eicu_lab_prediction_labtop.py --synthetic --eval-only \
  --checkpoint /tmp/labtop_demo/best_model.pt --max-new-tokens 8 \
  --history-keep 1 .75 .5 --output /tmp/labtop_reload
```

Saved checkpoints include architecture, split seed, item selection, tokenizer
vocabulary hash and patient-split-membership hash. The CLI reloads architecture and
seed and uses the checkpoint's adjacent tokenizer directory unless an explicit
`--tokenizer` is supplied. Incompatible configurations, splits, vocabularies or
selected items fail clearly. Membership hashes do **not** hash the raw data
values; retain the same dataset version and preprocessing when re-evaluating.

An older checkpoint containing just this adapter's `state_dict` is also supported.
Only the newly added `BaseModel` device marker may be absent. Other missing keys,
unexpected weights or incompatible shapes fail. For those legacy files, provide
all original architecture/tokenizer settings and preserve the original training
split yourself. The manifest labels their original split provenance unverified.
An arbitrary original-author or Lightning checkpoint is not automatically compatible.

## Use the group's clinical checkpoint

For eICU, with the group's original data, tokenizer and frozen split:

```bash
python examples/eicu_lab_prediction_labtop.py /path/to/eicu --eval-only \
  --checkpoint /path/to/best_model.pt --tokenizer /path/to/matching_tokenizer \
  --max-new-tokens 50 --history-keep 1 .75 .5 --output /path/to/new_evaluation
```

For a legacy weights-only file, also pass the original `--hidden-dim`, `--layers`,
`--heads`, `--max-seq-len`, `--seed` and loss-mode settings. If its original split
was created differently, supply `--split-file /path/to/original_patient_splits.json`.
That local JSON must contain exactly `train`, `valid`, and `test` lists of string
patient IDs, with every loaded patient assigned once and no overlap. The CLI
validates memberships and records only the file hash, never the IDs. A seed alone
does not recover a split made with a different algorithm or patient roster.
Reconcile files from the teammate's latest branch before assuming the checkpoint
matches this snapshot.
To train a new model, omit `--eval-only`; `--checkpoint` then warm-starts it.

For HiRID, use `--dataset hirid` and the documented raw-stage layout:
`general_table.csv`, `hirid_variable_reference.csv`,
`observation_tables/parquet/`, and `pharma_records/parquet/`.
The current processor's outcome set includes numeric observations such as vitals.
That broader task differs from the final paper's selected lab cohort.

Do not tune models, choose retention settings or choose checkpoints using test
scores. A separate validation experiment is needed for any such selection.

## Metric contract

For each item and unit, form valid numeric truth/prediction pairs first. Estimate
that group's 1st and 99th truth percentiles by linear interpolation and retain
inclusive boundaries. NMAE is mean absolute error divided by the percentile span.
SMAPE is `200 * |truth - prediction| / (|truth| + |prediction|)` in percent; a
zero/zero pair contributes zero. Aggregate group errors by retained counts.
This preserves the reference's defined-case formulas and ordering.

Report `numeric_predictions / prompts` and both metric denominators alongside
errors. Dropping invalid predictions can change percentile thresholds and the
scored subset. A lower error on a smaller numeric subset is not evidence of
better overall performance. Constant groups contribute to SMAPE only, and sparse
or empty groups yield undefined metrics. JSON represents undefined values as
`null`, and CSV leaves the corresponding cells blank.

`last_value` and `history_mean` use strictly earlier measurements of the same
item and unit. Missing history remains missing and is counted; it is not imputed
from test values. Their valid subsets can differ from the model's. For dropout
arms, historical baselines still describe the original full history and are
listed only once in `results.csv`. The paper describes a training-mean fallback
for first measurements; this port's baseline omission policy must be disclosed.

## New ablation: missing historical events

`--history-keep 1 .75 .5` evaluates a fixed checkpoint on the same target prompts.
Each historical event receives a seeded random draw. Matched seeds produce nested
retained sets across probabilities, preserving each kept event's full tokens and
chronological order. Target headers and answers never change. This probes missing
historical information without retraining or fitting on test data.

The paper already explores model configurations, context lengths, event types,
loss modes, numeric representations and time encodings. This experiment varies
random availability of **complete previous events**, rather than repeating those
published comparisons. It complements the group's smaller-model experiment.

Recorded results on **20 invented patients**, split 16 training / 2 validation /
2 test, with 24 held-out prompts, are retained under `results/synthetic/`:

| History retained | NMAE | SMAPE (%) | Numeric predictions | Retained for metrics |
|---|---:|---:|---:|---:|
| 100% | 0.3611 | 44.60 | 24/24 | 24 |
| 75% | 0.3788 | 47.97 | 22/24 | 22 |
| 50% | 0.2812 | 32.29 | 14/24 | 11 |

These measured values demonstrate the experiment and reporting flow only. At 50%
retention, the conditional errors fall while numeric coverage falls to 58.3%.
That is not evidence that deleting history improves the model. This run does not
supply clinical benchmark results or uncertainty across multiple training seeds.

## Output files

| Artifact | Contents |
|---|---|
| `results.csv` | Aggregate model/ablation and full-history baseline comparisons |
| `evaluation.json` | Per-item-and-unit errors, percentile limits and coverage counts |
| `training_history.json` | Training and validation token losses; empty in evaluation-only mode |
| `run_manifest.json` | Effective settings, counts, source hashes, environment, provenance and interpretation |
| `best_model.pt` | Best validation checkpoint and reconstruction metadata |
| `tokenizer/` | Matching extended tokenizer |

Patient identifiers and raw predictions are not written to these results.
Keep clinical datasets and trained clinical artifacts in the group's authorized
working environment; the public contribution can use aggregate results and the
invented-data demonstration.

## Known paper/source differences

The authors' source has an apparent staticmethod/signature inconsistency in its
metric evaluator and a prompt split based on a mask beginning at the lab name.
This port uses the intended numeric formulas and an explicit value boundary,
rather than duplicating those defects. Strict signed/range parsing, nonfinite
handling, unit-separated baselines and unrounded historical means are documented
robustness choices.

The current training mask includes item name, value, unit and delimiter, matching
the released mask. The default objective is full next-token AR; `--lab-only`
restricts it to that mask. The paper's wording describes a narrower value/unit/end
objective. This adapter also does not reconstruct all demographic or digit-level
text preprocessing from the paper. Changing those choices for a trained model
requires retraining or a separately justified compatibility procedure.

## Fast focused verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python -m pytest tests/core/test_labtop.py tests/core/test_labtop_model.py \
  tests/core/test_labtop_inference.py tests/core/test_labtop_metrics.py \
  tests/core/test_labtop_evaluation.py tests/core/test_labtop_splits.py \
  tests/core/test_hirid.py -q --durations=12
```

Model/task/evaluation tests use tiny tensors and invented patients with no network.
The two-patient HiRID fixture performs actual dataset loading once per module.
Its Dask/cache setup has several seconds of infrastructure overhead; report this
separately from the millisecond test calls instead of claiming zero-cost setup.
