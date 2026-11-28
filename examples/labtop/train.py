"""Train LabTOP and evaluate held-out HiRID prompts.

For HiRID 1.1.1 archive extraction, cache setup and logging, follow the guide
linked from examples/labtop/README.md. From the repository root, run one epoch
on an extracted HiRID directory with

    python -m examples.labtop.train /path/to/hirid --lab-only --epochs 1

Pass --tokenizer /path/to/Bio_ClinicalBERT for offline execution.
"""

import argparse
import json
import math
import random
from itertools import batched

import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pyhealth.data import Patient
from pyhealth.datasets import HIRIDDataset
from pyhealth.datasets.hirid import REQUIRED_TABLES
from pyhealth.metrics.labtop import labtop_metrics, parse_lab_value
from pyhealth.models import LabTOP
from pyhealth.tasks.labtop import (
    END, LabTOPHiRIDProcessor, collate_labtop, collate_labtop_prompts,
)


def iter_patient_batches(dataset, patient_ids, batch_size, stage):
    """Read several patients per parquet query and yield them in the requested order."""
    total = len(patient_ids)
    completed = 0
    print(f"{stage}: 0/{total} patients", flush=True)
    events = dataset.global_event_df
    for ids in batched(patient_ids, batch_size):
        frames = (
            events.filter(pl.col("patient_id").is_in(ids))
            .collect(engine="streaming")
            .partition_by("patient_id", as_dict=True)
        )
        for patient_id in ids:
            yield Patient(patient_id, frames.pop((patient_id,)))
        completed += len(ids)
        print(f"{stage}: {completed}/{total} patients", flush=True)


def train(model, train_loader, valid_loader, epochs, device, accumulation_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    steps = epochs * math.ceil(len(train_loader) / accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for index, batch in enumerate(train_loader):
            start = index - index % accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - start)
            loss = model(*(x.to(device) for x in batch))["loss"]
            (loss / group_size).backward()
            if index + 1 == start + group_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        model.eval()
        total_loss = count = 0
        with torch.no_grad():
            for batch in valid_loader:
                loss = model(*(x.to(device) for x in batch))["loss"]
                target_mask = batch[1] if model.lab_only else batch[0].ne(0)
                targets = target_mask[:, 1:].sum().item()
                total_loss += loss.item() * targets
                count += targets
        valid_loss = total_loss / count
        print(f"Epoch {epoch + 1} validation loss {valid_loss:.4f}", flush=True)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().to("cpu", copy=True)
                          for key, value in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= 10:
            break
    if not best_loss < float("inf"):
        raise ValueError("No finite validation loss")
    model.load_state_dict(best_state)


def evaluate(model, test_model_dataset, tokenizer, max_new_tokens, print_samples=False):
    model.eval()
    end = tokenizer.convert_tokens_to_ids(END)
    records = []
    with tqdm(test_model_dataset, desc="Evaluated", unit="prompt", disable=None) as progress:
        for row in progress:
            prompts, labels, _, _ = collate_labtop_prompts([row])
            ids = model.predict_value_tokens(prompts[0], end, max_new_tokens)
            text = tokenizer.decode(ids).replace(" ##", "")
            prediction = parse_lab_value(text, row["unit"])
            records.append((row["item"], row["lab_value"], prediction))
            if print_samples:
                sample = dict(sample=len(records), item=row["item"], unit=row["unit"],
                              prompt=tokenizer.decode(prompts[0]), expected=tokenizer.decode(labels[0]),
                              truth=row["lab_value"], generated=text, prediction=prediction)
                tqdm.write("Inference sample " + json.dumps(sample))
            if progress.disable:
                print(f"Evaluated {len(records)} held-out prompts", flush=True)
    return labtop_metrics(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--dataset", choices=["hirid"], default="hirid", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-dir",
        help="Parent directory of the HiRID event cache",
    )
    parser.add_argument("--patient-read-batch-size", type=int, default=32,
                        help="Patients per parquet read during preprocessing (default: 32)")
    parser.add_argument("--tokenizer", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument(
        "--lab-only", action=argparse.BooleanOptionalAction, default=True,
        help="Train and validate on numeric observation values, units and end markers "
             "(default). Use --no-lab-only for all-token training.",
    )
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--print-samples", action="store_true",
                        help="Print each decoded inference prompt, expected and generated text, and numeric prediction")
    args = parser.parse_args()
    if args.accumulation_steps < 1 or args.batch_size % args.accumulation_steps or args.batch_size < 1:
        parser.error("batch size must be a positive multiple of accumulation steps")
    if args.epochs < 1 or args.max_new_tokens < 1:
        parser.error("epochs and max-new-tokens must be positive")
    if args.patient_read_batch_size < 1:
        parser.error("patient-read-batch-size must be positive")
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = HIRIDDataset(args.root, list(REQUIRED_TABLES), cache_dir=args.cache_dir)
    processor = LabTOPHiRIDProcessor(tokenizer, dataset.variable_reference_path, args.max_seq_len)
    ids = sorted(dataset.unique_patient_ids)
    if len(ids) < 3:
        raise ValueError("Three distinct patients are required to form disjoint splits")
    random.Random(42).shuffle(ids)
    k = max(1, len(ids) // 10)
    test_ids, valid_ids = set(ids[:k]), set(ids[k:2 * k])
    print(f"Selected {len(ids)} patients: {len(ids) - 2*k} train, {k} validation, {k} test", flush=True)
    read_size = args.patient_read_batch_size
    processor.fit(iter_patient_batches(dataset, ids[2 * k:], read_size, "Fitting item frequencies"))
    train_dataset, valid_dataset = [], []
    for patient in iter_patient_batches(dataset, sorted(ids[k:]), read_size, "Encoding train/valid stays"):
        destination = valid_dataset if patient.patient_id in valid_ids else train_dataset
        destination.extend(processor.preprocess(patient, args.lab_only))
    if not train_dataset or not valid_dataset:
        raise ValueError("A split has no eligible stays or held-out lab prompts")
    print(f"{len(train_dataset)} train stays, {len(valid_dataset)} validation stays", flush=True)
    micro_batch_size = args.batch_size // args.accumulation_steps
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size,
                              collate_fn=processor.collate_train, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=micro_batch_size, collate_fn=collate_labtop)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LabTOP(tokenizer, args.max_seq_len, args.hidden_dim, args.layers, args.heads,
                   lab_only=args.lab_only).to(device)
    train(model, train_loader, valid_loader, args.epochs, device, args.accumulation_steps)
    test_patients = iter_patient_batches(dataset, sorted(test_ids), read_size, "Evaluating test stays")
    test_model_dataset = (row for patient in test_patients for row in processor.make_inference_dataset(patient))
    metrics = evaluate(model, test_model_dataset, tokenizer, args.max_new_tokens, print_samples=args.print_samples)
    if not metrics["prompts"]:
        raise ValueError("No eligible held-out laboratory prompts")
    print("item\tcount\tmin_v\tmax_v\tmae\tnmae\tsmape")
    for row in metrics.pop("per_item"):
        print("{item}\t{count}\t{min_v:.6g}\t{max_v:.6g}\t{mae:.6g}\t{nmae:.6g}\t{smape:.6g}".format(**row))
    print(metrics)


if __name__ == "__main__":
    main()
