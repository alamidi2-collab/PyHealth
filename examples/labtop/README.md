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
