"""Train LabTOP and evaluate genuinely held-out eICU or HiRID prompts.

The offline synthetic mode exercises the complete PyHealth flow without a
clinical dataset or tokenizer download. Clinical inference uses local data and a
local checkpoint. Saved results never contain patient IDs or raw predictions.
See examples/labtop/README.md for commands, metric definitions and limitations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import tempfile
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from pyhealth.datasets import HIRIDDataset, create_sample_dataset, eICUDataset
from pyhealth.datasets.hirid import REQUIRED_TABLES
from pyhealth.metrics.labtop import labtop_metrics, parse_lab_value
from pyhealth.models import LabTOP
from pyhealth.tasks.labtop import (
    END,
    LabTOPHiRIDProcessor,
    LabTOPProcessor,
    LabTOPTask,
    collate_labtop,
    collate_labtop_prompts,
)

REFERENCE_COMMIT = "fa2e66749af84338f6cef7ba3c5918e09719c0e4"


def json_safe(value: Any) -> Any:
    """Represent undefined metrics as JSON null rather than nonstandard NaN."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Write strictly valid, human-readable JSON."""
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def split_patient_ids(patient_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    """Create stable disjoint splits, keeping every stay of a patient together.

    The 80/10/10 split uses integer rounding. For small synthetic inputs each
    held-out partition gets at least one patient, so its ratios can differ.
    """
    ids = sorted(set(patient_ids))
    if len(ids) < 3:
        raise ValueError(
            "At least three distinct patients are needed for disjoint splits"
        )
    random.Random(seed).shuffle(ids)
    k = max(1, len(ids) // 10)
    return {"train": ids[2 * k :], "valid": ids[k : 2 * k], "test": ids[:k]}


def supervised_count(batch: tuple[torch.Tensor, ...], lab_only: bool) -> int:
    """Count real next-token targets, excluding padding and the first token."""
    keep = batch[0][:, 1:].ne(0)
    if lab_only:
        keep = keep & batch[1][:, 1:].bool()
    return int(keep.sum())


def train(
    model: LabTOP,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: torch.device,
    accumulation_steps: int,
    *,
    learning_rate: float = 1e-4,
    patience: int = 10,
) -> list[dict[str, float | int]]:
    """Train using validation loss only and restore the best checkpoint.

    Accumulation weights microbatches by their supervised token counts, so
    different sequence lengths do not silently change the virtual-batch loss.
    Test prompts and test metrics never influence optimization or selection.
    """
    if epochs < 1 or accumulation_steps < 1 or patience < 1:
        raise ValueError("epochs, accumulation_steps and patience must be positive")
    if not len(train_loader) or not len(valid_loader):
        raise ValueError("Training and validation loaders must both be nonempty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps = epochs * math.ceil(len(train_loader) / accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    best_loss, best_epoch, best_state = math.inf, 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        iterator = iter(train_loader)
        train_total = train_count = 0
        while group := list(islice(iterator, accumulation_steps)):
            counts = [supervised_count(batch, model.lab_only) for batch in group]
            target_count = sum(counts)
            if not target_count:
                continue
            optimizer.zero_grad(set_to_none=True)
            for batch, count in zip(group, counts):
                if not count:
                    continue
                loss = model(*(x.to(device) for x in batch))["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became nonfinite")
                (loss * count / target_count).backward()
                train_total += float(loss.detach()) * count
                train_count += count
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Training gradients became nonfinite")
            optimizer.step()
            scheduler.step()
        if not train_count:
            raise ValueError("Training split contains no supervised next-token targets")
        model.eval()
        total_loss = count = 0
        with torch.no_grad():
            for batch in valid_loader:
                targets = supervised_count(batch, model.lab_only)
                if targets:
                    loss = model(*(x.to(device) for x in batch))["loss"]
                    total_loss += float(loss) * targets
                    count += targets
        if not count:
            raise ValueError(
                "Validation split contains no supervised next-token targets"
            )
        valid_loss = total_loss / count
        if not math.isfinite(valid_loss):
            raise RuntimeError("Validation loss became nonfinite")
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_total / train_count,
                "valid_loss": valid_loss,
                "valid_tokens": count,
            }
        )
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            best_state = {
                k: v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    return history


def evaluate(
    model: LabTOP,
    test_prompt_dataset: list[dict[str, Any]],
    tokenizer: Any,
    max_new_tokens: int = 50,
    *,
    return_details: bool = False,
) -> dict[str, Any]:
    """Recreate the authors' inference-to-item-metrics calculation.

    Only historical context plus target time/table/item enters the model.
    Truths remain in the metric records. Numeric parse coverage accompanies
    conditional errors. Last-value and history-mean baselines are separate;
    missing same-unit history stays missing, with no test-derived imputation.
    """
    end = tokenizer.convert_tokens_to_ids(END)
    records: dict[str, list[tuple[Any, float, float | None]]] = {
        "model": [],
        "last_value": [],
        "history_mean": [],
    }
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for row in test_prompt_dataset:
            prompts, _, _, _ = collate_labtop_prompts([row])
            ids = model.predict_value_tokens(prompts[0], end, max_new_tokens)
            text = tokenizer.decode(ids, skip_special_tokens=False).replace(" ##", "")
            key = (row["item"], row["unit"])
            truth = row["lab_value"]
            values = {
                "model": parse_lab_value(text, row["unit"]),
                "last_value": row.get("previous_value"),
                "history_mean": row.get("mean_previous_values"),
            }
            for name, value in values.items():
                records[name].append((key, truth, value))
    finally:
        for module, mode in modes.items():
            module.training = mode
    if not records["model"]:
        raise ValueError("The held-out test split has no eligible laboratory prompts")
    results = {
        name: labtop_metrics(rows, return_per_item=return_details)
        for name, rows in records.items()
    }
    for result in results.values():
        result["numeric_coverage"] = result["numeric_predictions"] / result["prompts"]
    return results if return_details else results["model"]


def ablate_history(
    row: dict[str, Any], keep_probability: float, seed: int, end_token_id: int
) -> dict[str, Any]:
    """Remove whole historical events while preserving the target and its header.

    Identical seeds give nested retained-event sets across probabilities. This
    inference robustness experiment requires no tuning or retraining on test data.
    Baseline values in metadata still describe the original full history.
    """
    if not 0 <= keep_probability <= 1:
        raise ValueError("History retention probabilities must be in [0, 1]")
    events, start = [], 0
    tokens = row["prompt_tokens"]
    for i, token in enumerate(tokens):
        if token == end_token_id:
            events.append(tokens[start : i + 1])
            start = i + 1
    if start != len(tokens):
        raise ValueError("Historical context must contain only complete events")
    rng = random.Random(seed)
    kept = [event for event in events if rng.random() < keep_probability]
    return {
        **row,
        "prompt_tokens": [token for event in kept for token in event],
        "history_events": len(events),
        "retained_history_events": len(kept),
    }


def load_model_weights(model: LabTOP, payload: dict[str, Any]) -> None:
    """Load a saved state, allowing only the newly added BaseModel device marker.

    Existing tokenizer-first LabTOP checkpoints retain all transformer parameter
    names. Any other missing/unexpected weight or shape mismatch is an error.
    """
    state = payload.get("model_state_dict", payload)
    result = model.load_state_dict(state, strict=False)
    missing = set(result.missing_keys) - {"_dummy_param"}
    if missing or result.unexpected_keys:
        raise ValueError(
            f"Checkpoint does not match LabTOP: missing={sorted(missing)}, "
            f"unexpected={result.unexpected_keys}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI options without loading a dataset or tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-patients", type=int, default=20)
    parser.add_argument("--dataset", choices=["eicu", "hirid"], default="eicu")
    parser.add_argument("--tables", nargs="+")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--tokenizer", help="Local tokenizer directory or HF identifier"
    )
    parser.add_argument("--lab-only", action="store_true", default=None)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--split-file",
        type=Path,
        help="JSON with original train/valid/test patient IDs",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--history-keep", type=float, nargs="+", default=[1.0])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("labtop_run"))
    return parser


def run(args: argparse.Namespace, root: Path, tokenizer_path: str) -> dict[str, Any]:
    """Execute one fixed split, one trained checkpoint, and paired history arms."""
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if args.dataset == "hirid":
        tables = args.tables or list(REQUIRED_TABLES)
        dataset = HIRIDDataset(root, tables, cache_dir=args.cache_dir)
        processor = LabTOPHiRIDProcessor(
            tokenizer, dataset.variable_reference_path, args.max_seq_len
        )
    else:
        tables = args.tables or ["lab", "medication", "treatment"]
        dataset = eICUDataset(str(root), tables, cache_dir=args.cache_dir)
        processor = LabTOPProcessor(tokenizer, args.max_seq_len)
    if args.split_file:
        from examples.labtop.splits import load_patient_splits

        splits = load_patient_splits(args.split_file, dataset.unique_patient_ids)
    else:
        splits = split_patient_ids(dataset.unique_patient_ids, args.seed)
    cohort_sha256 = hashlib.sha256(
        json.dumps(
            {part: sorted(ids) for part, ids in splits.items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    tokenizer_sha256 = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    processor.fit(dataset.get_patient(i) for i in splits["train"])
    task = LabTOPTask(processor, lab_only=args.lab_only)
    train_rows = [row for i in splits["train"] for row in task(dataset.get_patient(i))]
    valid_rows = [row for i in splits["valid"] for row in task(dataset.get_patient(i))]
    if not train_rows or not valid_rows:
        raise ValueError(
            "Training or validation split has no eligible supervised stays"
        )
    train_dataset = create_sample_dataset(
        samples=train_rows,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
    )
    valid_dataset = create_sample_dataset(
        samples=valid_rows,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
    )
    config = {
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "lab_only": args.lab_only,
        "dropout": args.dropout,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LabTOP(tokenizer, dataset=train_dataset, **config).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("cohort_sha256", cohort_sha256) != cohort_sha256:
            raise ValueError("Checkpoint patient splits differ from this dataset")
        if payload.get("tokenizer_sha256", tokenizer_sha256) != tokenizer_sha256:
            raise ValueError("Checkpoint tokenizer vocabulary differs")
        expected_source = "synthetic" if args.synthetic else args.dataset
        if payload.get("data_source", expected_source) != expected_source:
            raise ValueError("Checkpoint data source differs from requested source")
        if (
            "selected_items" in payload
            and set(map(tuple, payload["selected_items"])) != processor.selected_items
        ):
            raise ValueError("Training item selection differs from the checkpoint")
        load_model_weights(model, payload)
    micro = args.batch_size // args.accumulation_steps
    history = []
    if not args.eval_only:
        train_loader = DataLoader(
            Subset(train_dataset, range(len(train_dataset))),
            batch_size=micro,
            collate_fn=processor.collate_train,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        valid_loader = DataLoader(
            Subset(valid_dataset, range(len(valid_dataset))),
            batch_size=micro,
            collate_fn=collate_labtop,
        )
        history = train(
            model,
            train_loader,
            valid_loader,
            args.epochs,
            device,
            args.accumulation_steps,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )
    # Explicitly create TEST prompts after validation selection. Validation rows
    # are never reused as a reported held-out test prompt dataset.
    test_prompt_dataset = [
        row
        for i in splits["test"]
        for row in processor.make_inference_dataset(
            dataset.get_patient(i), max_new_tokens=args.max_new_tokens
        )
    ]
    if not test_prompt_dataset:
        raise ValueError("Test split has no eligible held-out laboratory prompts")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output / "tokenizer")
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "model_config": config,
            "split_seed": args.seed,
            "data_source": "synthetic" if args.synthetic else args.dataset,
            "selected_items": sorted(processor.selected_items),
            "cohort_sha256": cohort_sha256,
            "tokenizer_sha256": tokenizer_sha256,
        },
        args.output / "best_model.pt",
    )
    end = tokenizer.convert_tokens_to_ids(END)
    evaluations, rows = {}, []
    for keep in sorted(set(args.history_keep), reverse=True):
        prompts = [
            ablate_history(row, keep, args.seed + j, end)
            for j, row in enumerate(test_prompt_dataset)
        ]
        result = evaluate(
            model, prompts, tokenizer, args.max_new_tokens, return_details=True
        )
        evaluations[str(keep)] = result
        for method, metrics in result.items():
            # History baselines have full original history and are listed once.
            if method != "model" and keep != 1.0:
                continue
            rows.append(
                {
                    "data_source": "synthetic" if args.synthetic else args.dataset,
                    "method": method,
                    "history_keep": keep,
                    **{k: v for k, v in metrics.items() if k != "per_item"},
                }
            )
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_safe(rows))
    write_json(args.output / "evaluation.json", evaluations)
    write_json(args.output / "training_history.json", history)
    repo = Path(__file__).resolve().parents[2]
    relevant = [
        "pyhealth/models/labtop.py",
        "pyhealth/tasks/labtop.py",
        "pyhealth/metrics/labtop.py",
        "examples/labtop/train.py",
        "examples/labtop/synthetic.py",
        "examples/labtop/splits.py",
    ]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_source": "synthetic" if args.synthetic else args.dataset,
        "synthetic_only": args.synthetic,
        "reference_commit": REFERENCE_COMMIT,
        "model_config": config,
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "accumulation_steps": args.accumulation_steps,
        },
        "split_seed": args.seed,
        "split_origin": "explicit_patient_memberships"
        if args.split_file
        else "seeded_80_10_10",
        "split_file_sha256": hashlib.sha256(args.split_file.read_bytes()).hexdigest()
        if args.split_file
        else None,
        "split_patient_counts": {k: len(v) for k, v in splits.items()},
        "split_stay_counts": {"train": len(train_rows), "valid": len(valid_rows)},
        "test_prompt_count": len(test_prompt_dataset),
        "history_keep": sorted(set(args.history_keep), reverse=True),
        "max_new_tokens": args.max_new_tokens,
        "tables": tables,
        "synthetic_patients": args.synthetic_patients if args.synthetic else None,
        "evaluation_only": args.eval_only,
        "cohort_sha256": cohort_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "checkpoint_provenance": (
            "recorded_split_and_vocabulary"
            if not args.checkpoint
            or ("cohort_sha256" in payload and "tokenizer_sha256" in payload)
            else "legacy_weights_only_original_training_split_unverified"
        ),
        "source_checkpoint_sha256": (
            hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if args.checkpoint
            else None
        ),
        "best_validation_epoch": (
            min(history, key=lambda row: row["valid_loss"])["epoch"]
            if history
            else None
        ),
        "checkpoint_sha256": hashlib.sha256(
            (args.output / "best_model.pt").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in relevant
        },
        "environment": {
            "python": platform.python_version(),
            "device": str(device),
            "threads": args.threads,
            **{
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "transformers",
                    "numpy",
                    "pandas",
                    "polars",
                    "pyhealth",
                )
            },
        },
        "interpretation": (
            "Synthetic execution demonstration only; no clinical performance claim."
            if args.synthetic
            else "Local clinical run. Confirm cohort, tokenization and split parity "
            "before comparing with published results."
        ),
        "metric_notes": "Item-and-unit groups. Conditional errors exclude invalid "
        "predictions; compare numeric_coverage too. Baseline errors "
        "omit missing same-unit history; no test-derived imputation.",
    }
    write_json(args.output / "run_manifest.json", manifest)
    print(
        json.dumps(
            json_safe(
                {
                    "data_source": manifest["data_source"],
                    "test_prompts": len(test_prompt_dataset),
                    "results": rows,
                }
            ),
            indent=2,
        )
    )
    return manifest


def main() -> None:
    """Validate arguments and run either a local-data or offline synthetic flow."""
    parser = build_parser()
    args = parser.parse_args()
    if args.synthetic and (args.root or args.dataset != "eicu"):
        parser.error("--synthetic supplies its own invented eICU root")
    if not args.synthetic and args.root is None:
        parser.error("Provide a local dataset root or use --synthetic")
    if args.eval_only and args.checkpoint is None:
        parser.error("--eval-only requires --checkpoint")
    saved = {}
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        saved = payload.get("model_config", {})
        saved_seed = payload.get("split_seed")
        if saved_seed is not None and args.seed not in (None, saved_seed):
            parser.error(
                "Checkpoint split seed differs; preserve the original held-out split"
            )
        args.seed = saved_seed if args.seed is None else args.seed
    args.seed = 42 if args.seed is None else args.seed
    defaults = {
        "max_seq_len": 128 if args.synthetic else 4096,
        "hidden_dim": 32 if args.synthetic else 512,
        "layers": 1 if args.synthetic else 12,
        "heads": 2 if args.synthetic else 8,
        "lab_only": False,
        "dropout": 0.1,
    }
    for name, default in defaults.items():
        key = {"layers": "n_layers", "heads": "n_heads"}.get(name, name)
        value = getattr(args, name)
        if key in saved and value is not None and value != saved[key]:
            parser.error(
                f"--{name.replace('_', '-')} conflicts with checkpoint configuration"
            )
        setattr(args, name, saved.get(key, default) if value is None else value)
    if (
        args.accumulation_steps < 1
        or args.batch_size < 1
        or (args.batch_size % args.accumulation_steps)
    ):
        parser.error("batch size must be a positive multiple of accumulation steps")
    if min(args.epochs, args.max_new_tokens, args.threads, args.patience) < 1:
        parser.error("epochs, generation budget, threads and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if any(not 0 <= x <= 1 for x in args.history_keep) or len(
        set(args.history_keep)
    ) != len(args.history_keep):
        parser.error("history retention probabilities must be unique and in [0, 1]")
    if 1.0 not in args.history_keep:
        parser.error("Include --history-keep 1 to retain the full-history control")
    if args.max_new_tokens >= args.max_seq_len:
        parser.error("Generation budget must be smaller than max-seq-len")
    if args.synthetic_patients < 3:
        parser.error("Synthetic data need at least three distinct patients")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a fresh --output directory to preserve previous results")
    local_tokenizer = args.checkpoint.parent / "tokenizer" if args.checkpoint else None
    tokenizer_path = args.tokenizer or (
        str(local_tokenizer) if local_tokenizer and local_tokenizer.is_dir() else None
    )
    if args.synthetic:
        from examples.labtop.synthetic import write_synthetic_eicu

        with tempfile.TemporaryDirectory(prefix="labtop_synthetic_") as directory:
            root = Path(directory)
            vocab = write_synthetic_eicu(root, args.synthetic_patients, args.seed)
            args.cache_dir = root / "cache"
            run(args, root, tokenizer_path or str(vocab))
    else:
        run(args, args.root, tokenizer_path or "emilyalsentzer/Bio_ClinicalBERT")


if __name__ == "__main__":
    main()
