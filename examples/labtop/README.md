# Train LabTOP on HiRID

Follow [Running the HiRID example](../../docs/api/models/pyhealth.models.LabTOP.rst#running-the-hirid-example) for the commands to extract the HiRID 1.1.1 archives, run training and held-out inference, and save the console output to a log.

If the raw parquet files and reference tables are already extracted, the training entry point is available from the PyHealth repository root:

```bash
python -m examples.labtop.train /path/to/hirid --lab-only --epochs 1
```

Evaluation shows a running count of completed prompts. With redirected output
(or `tee`), each completed prompt gets a progress line. Add `--print-samples`
to also log each decoded input prompt, item, unit, expected completion, ground
truth, generated text, and parsed prediction as a numbered JSON record.

The final `uncertainty` report gives a pooled Spearman correlation and sample
count. For each parsed completion, the logged `uncertainty_score` is the mean full-vocabulary entropy
(in nats) at the generated steps covering the numeric text. Each error is the
absolute prediction error divided by that item's 99th minus 1st percentile
of observed truths among numeric pairs. The correlation uses observations
before percentile trimming and excludes parse failures, exact matches and
zero-span items. A constant rank variable or fewer than two usable pairs gives
`nan`. Positive rho indicates that higher uncertainty ranks with larger errors.
