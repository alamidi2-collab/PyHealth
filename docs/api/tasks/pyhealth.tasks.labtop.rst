LabTOP sequence tasks and inference prompts
==========================================

These components implement the eICU and HiRID sequence interfaces for
`LabTOP: A Unified Model for Lab Test Outcome Prediction on Electronic Health
Records <https://proceedings.mlr.press/v287/im25a.html>`_ (Im, Oh and Choi,
CHIL 2025). The
`authors' implementation <https://github.com/sujeongim/LabTOP>`_ is the reference
for serialization, complete-event permutation and prompt-based evaluation.

Training integration
--------------------

``LabTOPProcessor`` reads existing PyHealth eICU patient events.
``LabTOPHiRIDProcessor`` reads HiRID observations and medication records using
the variable-reference file. Both retain complete events and produce one
variable-length token sequence per eligible ICU stay. Stays shorter than six
hours are excluded. The ``types`` field marks laboratory item/value/unit/end
positions for the optional laboratory-only next-token loss.

``LabTOPTask`` inherits :class:`~pyhealth.tasks.BaseTask` and adapts a fitted
processor without changing its token IDs or masks. Fit the item-frequency
selection on training patients only, then apply the same processor to every
split. The task never refits on patients it is asked to process.

.. code-block:: python

    from pyhealth.datasets import create_sample_dataset
    from pyhealth.tasks import LabTOPProcessor, LabTOPTask
    from pyhealth.tasks.labtop import collate_labtop
    from torch.utils.data import DataLoader

    processor = LabTOPProcessor(tokenizer, max_seq_len=4096)
    processor.fit(train_patients)
    task = LabTOPTask(processor, lab_only=True)
    rows = [row for patient in train_patients for row in task(patient)]
    samples = create_sample_dataset(rows, task.input_schema, task.output_schema)
    train_loader = DataLoader(
        samples, batch_size=8, collate_fn=processor.collate_train
    )
    # Validation uses collate_labtop, without random event permutation.

Keep ``train_patients`` reusable (for example a list); fitting consumes an
iterator. The schemas use raw processors because token IDs have already been
assigned and sequence lengths differ. The model creates shifted causal labels
from ``tokens``; ``types`` is a supervision mask, not a clinical class label.

Held-out inference
------------------

Select disjoint test patients in the caller. ``make_inference_dataset`` does
not make splits and does not fit vocabulary selection. It yields each numeric
lab target with its preceding context and separate target tokens.

.. code-block:: python

    from pyhealth.tasks.labtop import collate_labtop_prompts

    rows = [
        row
        for patient in test_patients
        for row in processor.make_inference_dataset(patient, max_new_tokens=50)
    ]
    prefix, target_tokens, previous_tokens, historical_means = (
        collate_labtop_prompts(rows[:2])
    )

The prompt contains only complete events from strictly earlier ten-minute
bins, followed by the requested test's time, table and item name. Every event
in the target's bin is excluded, including measurements recorded a few
minutes before the target. The value, unit and end marker are separate labels.
Targets without enough space for one complete historical event are skipped.

Pass the same ``max_new_tokens`` used for generation. Prompt truncation
reserves that fixed budget without inspecting the target answer's length.
Omitting it preserves the original small-example default of
``min(50, max_seq_len // 4)`` (at least one token).

Baselines and deviations from the reference
------------------------------------------

``previous_value`` and ``mean_previous_values`` are numerical baselines from
strictly earlier observations of the same item and measurement unit. The
original ``previous_value_tokens`` field and four-tensor collator return value
remain available. Baselines can use earlier observations that no longer fit in
the model's context window. Missing baseline history is represented by NaN and
an empty token list; evaluators must report its coverage explicitly.

Two deliberate robustness changes differ from the released reference code:

* Baselines never combine values recorded in different units. Unit spelling
  is normalized only for whitespace at its ends and case; no unit conversion
  is inferred.
* Historical means are calculated from the unrounded running sum. The
  reference recursively rounds updates to two decimals, which can accumulate
  rounding error for small values.

These changes concern baseline calculation, not the definition of model
inputs. A synthetic execution check does not establish real-data reproduction
accuracy. Use the experiment scripts and record data splits, tokenizer,
configuration and held-out metrics for a real evaluation.

API
---

.. autoclass:: pyhealth.tasks.labtop.LabTOPTask
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.labtop.LabTOPProcessor
    :members:
    :inherited-members:

.. autoclass:: pyhealth.tasks.labtop.LabTOPHiRIDProcessor
    :members:
    :inherited-members:

.. autofunction:: pyhealth.tasks.labtop.collate_labtop

.. autofunction:: pyhealth.tasks.labtop.collate_labtop_prompts
