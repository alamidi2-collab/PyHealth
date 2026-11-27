"""Train LabTOP and evaluate held-out HiRID prompts.

Train a HuggingFace GPT-2 model on laboratory prompts from the HiRID dataset (ver 1.1.1 of HiRID dataset required from PhysioNet)

From the repository root, run e.g.
python -m examples.labtop.train /path/to/hirid --lab-only \
    --epochs 15 --max-seq-len 2048 --hidden-dim 512 --layers 12 --heads 8 --batch-size 32

Pass --tokenizer /path/to/Bio_ClinicalBERT for offline execution.
"""

import argparse
import math
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pyhealth.datasets import HIRIDDataset
from pyhealth.datasets.hirid import REQUIRED_TABLES
from pyhealth.metrics.labtop import labtop_metrics, parse_lab_value
from pyhealth.models import LabTOP
from pyhealth.tasks.labtop import (
    END, LabTOPHiRIDProcessor, collate_labtop, collate_labtop_prompts,
)


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


def evaluate(model, test_model_dataset, tokenizer, max_new_tokens):
    model.eval()
    end = tokenizer.convert_tokens_to_ids(END)
    records = []
    for row in test_model_dataset:
        prompts, _, _, _ = collate_labtop_prompts([row])
        ids = model.predict_value_tokens(prompts[0], end, max_new_tokens)
        text = tokenizer.decode(ids).replace(" ##", "")
        records.append((row["item"], row["lab_value"], parse_lab_value(text, row["unit"])))
    return labtop_metrics(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--dataset", choices=["hirid"], default="hirid", help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir")
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
    args = parser.parse_args()
    if args.accumulation_steps < 1 or args.batch_size % args.accumulation_steps or args.batch_size < 1:
        parser.error("batch size must be a positive multiple of accumulation steps")
    if args.epochs < 1 or args.max_new_tokens < 1:
        parser.error("epochs and max-new-tokens must be positive")
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
    processor.fit(dataset.get_patient(i) for i in ids[2 * k:])
    train_dataset, valid_dataset = [], []
    for patient_id in sorted(ids[k:]):
        destination = valid_dataset if patient_id in valid_ids else train_dataset
        destination.extend(processor.preprocess(dataset.get_patient(patient_id), args.lab_only))
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
    test_model_dataset = (row for i in sorted(test_ids)
                          for row in processor.make_inference_dataset(dataset.get_patient(i)))
    metrics = evaluate(model, test_model_dataset, tokenizer, args.max_new_tokens)
    if not metrics["prompts"]:
        raise ValueError("No eligible held-out laboratory prompts")
    print(metrics)


if __name__ == "__main__":
    main()
