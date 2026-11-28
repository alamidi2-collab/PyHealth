# Train LabTOP on HiRID

Follow [Running the HiRID example](../../docs/api/models/pyhealth.models.LabTOP.rst#running-the-hirid-example) for the commands to extract the HiRID 1.1.1 archives, run training and held-out inference, and save the console output to a log.

If the raw parquet files and reference tables are already extracted, the training entry point is available from the PyHealth repository root:

```bash
python -m examples.labtop.train /path/to/hirid --lab-only --epochs 1
```
