"""Fast offline LabTOP model checks using two short synthetic token sequences."""

import math
from unittest.mock import patch

import pytest
import torch
from transformers import BertTokenizerFast

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import BaseModel, LabTOP


@pytest.fixture
def tokenizer(tmp_path):
    """Use a local twelve-token vocabulary; never contact a model hub."""
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "lab",
        "glucose",
        "1",
        "2",
        "3",
        "mg",
        "|endofevent|",
    ]
    (tmp_path / "vocab.txt").write_text("\n".join(tokens), encoding="utf-8")
    return BertTokenizerFast.from_pretrained(tmp_path, local_files_only=True)


@pytest.fixture
def batch():
    return {
        "input_ids": torch.tensor([[5, 6, 7, 10, 11, 0], [5, 6, 8, 10, 9, 11]]),
        "train_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5]]),
    }


@pytest.fixture
def model(tokenizer):
    with torch.random.fork_rng():
        torch.manual_seed(11)
        return LabTOP(tokenizer, 16, 8, 1, 2, dropout=0.0, lab_only=True)


def test_base_model_with_optional_native_dataset(tokenizer, batch):
    samples = [
        {
            "patient_id": f"synthetic_{index}",
            **{key: value[index].tolist() for key, value in batch.items()},
        }
        for index in range(2)
    ]
    dataset = create_sample_dataset(
        samples,
        input_schema={key: "raw" for key in batch},
        output_schema={},
    )
    model = LabTOP(tokenizer, 16, 8, 1, 2, dropout=0.0, dataset=dataset)
    assert isinstance(model, BaseModel)
    assert model.dataset is dataset
    assert model.feature_keys == list(batch)
    assert model.label_processors == {}
    assert model.mode is None
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("lab_only", [False, True])
def test_forward_shapes_shifted_loss_padding_and_backward(model, batch, lab_only):
    model.lab_only = lab_only
    output = model(**batch)
    logits = output["logits"]
    assert logits.shape == (2, 6, 12)
    assert output["loss"].shape == ()
    # Independent per-token log probabilities check both causal shift and masks.
    losses = []
    log_probs = logits.log_softmax(dim=-1)
    for row in range(2):
        for target_position in range(1, 6):
            target = int(batch["input_ids"][row, target_position])
            supervised = bool(batch["train_mask"][row, target_position])
            if target != 0 and (supervised or not lab_only):
                losses.append(-log_probs[row, target_position - 1, target])
    torch.testing.assert_close(output["loss"], torch.stack(losses).mean())
    output["loss"].backward()
    gradient = model.model.transformer.wte.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0

    # Appending masked right padding must not alter supervised logits or loss.
    padded = {
        key: torch.cat([value, torch.zeros(2, 2, dtype=value.dtype)], dim=1)
        for key, value in batch.items()
    }
    padded_output = model(**padded)
    torch.testing.assert_close(output["loss"], padded_output["loss"])
    torch.testing.assert_close(logits[:, :5], padded_output["logits"][:, :5])


def test_embedding_forward_matches_tokens_and_has_input_gradients(model, batch):
    model.eval()
    expected = model(**batch)
    embeddings = (
        model.model.get_input_embeddings()(batch["input_ids"])
        .detach()
        .requires_grad_(True)
    )
    actual = model.forward_from_embedding(embeddings, **batch)
    torch.testing.assert_close(actual["logits"], expected["logits"])
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.count_nonzero(embeddings.grad) > 0


@pytest.mark.parametrize("lab_only", [False, True])
def test_zero_supervised_tokens_has_finite_differentiable_zero(model, batch, lab_only):
    model.lab_only = lab_only
    batch["train_mask"].zero_()
    if not lab_only:
        batch["input_ids"][:, 1:] = 0
    output = model(**batch)
    assert output["loss"].requires_grad
    assert output["loss"].item() == 0.0
    output["loss"].backward()
    gradient = model.model.transformer.wte.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


def test_malformed_training_inputs_fail_clearly(model, batch):
    invalid = [
        ("input_ids", torch.ones(6, dtype=torch.long)),
        ("input_ids", torch.empty(0, 6, dtype=torch.long)),
        ("input_ids", torch.ones(2, 1, dtype=torch.long)),
        ("input_ids", torch.ones(2, 17, dtype=torch.long)),
        ("input_ids", batch["input_ids"].float()),
        ("input_ids", torch.full((2, 6), -1, dtype=torch.long)),
        ("input_ids", torch.full((2, 6), 12, dtype=torch.long)),
        ("train_mask", torch.ones(2, 5)),
        ("train_mask", torch.full((2, 6), 0.5)),
        ("train_mask", torch.full((2, 6), math.nan)),
        ("position_ids", batch["position_ids"].float()),
        ("position_ids", torch.full((2, 6), -1, dtype=torch.long)),
        ("position_ids", torch.full((2, 6), 16, dtype=torch.long)),
    ]
    for key, value in invalid:
        with pytest.raises(ValueError):
            model(**{**batch, key: value})


def test_malformed_embeddings_fail_clearly(model, batch):
    for value in (torch.ones(2, 6, 7), torch.full((2, 6, 8), math.nan)):
        with pytest.raises(ValueError, match="embeddings"):
            model.forward_from_embedding(value, **batch)


@pytest.mark.parametrize("mode", ["train", "eval", "mixed"])
def test_generation_stops_at_event_and_restores_every_mode(model, mode):
    model.train(mode != "eval")
    if mode == "mixed":
        model.model.transformer.drop.eval()
    previous_modes = {module: module.training for module in model.modules()}
    previous_state = {name: value.clone() for name, value in model.state_dict().items()}

    def generate(input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 3
        assert kwargs["eos_token_id"] == 11
        assert kwargs["pad_token_id"] == 0
        assert torch.equal(kwargs["attention_mask"], torch.ones_like(input_ids))
        return torch.cat([input_ids, torch.tensor([[7, 11]])], dim=1)

    with patch.object(model.model, "generate", side_effect=generate):
        assert model.predict_value_tokens([5, 6], 11, 3) == [7]
    assert {module: module.training for module in model.modules()} == previous_modes
    for name, value in model.state_dict().items():
        assert torch.equal(value, previous_state[name])
    assert torch.is_grad_enabled()


def test_generation_restores_modes_after_exception(model):
    model.train()
    model.model.transformer.drop.eval()
    previous_modes = {module: module.training for module in model.modules()}
    with (
        patch.object(model.model, "generate", side_effect=RuntimeError("test failure")),
        pytest.raises(RuntimeError, match="test failure"),
    ):
        model.predict_value_tokens([5, 6], 11, 2)
    assert {module: module.training for module in model.modules()} == previous_modes
    assert torch.is_grad_enabled()


def test_immediate_delimiter_and_context_limit(model):
    def generate(input_ids, **kwargs):
        assert kwargs["max_new_tokens"] == 1
        return torch.cat([input_ids, torch.tensor([[11]])], dim=1)

    with patch.object(model.model, "generate", side_effect=generate):
        assert model.predict_value_tokens([5] * 15, 11, 50) == []


def test_invalid_generation_arguments(model):
    bad_calls = [
        ([], 11, 2),
        ([[5, 6]], 11, 2),
        ([5.0, 6.0], 11, 2),
        ([0, 5], 11, 2),
        ([-1, 5], 11, 2),
        ([12, 5], 11, 2),
        ([5] * 16, 11, 2),
        ([5, 6], 0, 2),
        ([5, 6], 12, 2),
        ([5, 6], True, 2),
        ([5, 6], 11, 0),
        ([5, 6], 11, True),
        ([5, 6], 11, 1.5),
    ]
    with patch.object(model.model, "generate") as generate:
        for prompt, end, budget in bad_calls:
            with pytest.raises(ValueError):
                model.predict_value_tokens(prompt, end, budget)
        generate.assert_not_called()


def test_actual_tiny_generation_smoke(model):
    model.train()
    output = model.predict_value_tokens([5, 6], 11, 2)
    assert isinstance(output, list)
    assert len(output) <= 2
    assert all(isinstance(token, int) and 0 <= token < 12 for token in output)
    assert 11 not in output
    assert model.training


def test_original_checkpoint_keys_unchanged_except_base_model_dummy(model, tokenizer):
    state = model.state_dict()
    old_state = {
        key: value.clone() for key, value in state.items() if key != "_dummy_param"
    }
    assert all(key.startswith("model.") for key in old_state)
    restored = LabTOP(tokenizer, 16, 8, 1, 2, dropout=0.0)
    missing, unexpected = restored.load_state_dict(old_state, strict=False)
    assert missing == ["_dummy_param"]
    assert unexpected == []
    for key, value in old_state.items():
        assert torch.equal(restored.state_dict()[key], value)


@pytest.mark.parametrize(
    "config",
    [
        {"max_seq_len": 1},
        {"hidden_dim": 0},
        {"n_layers": 0},
        {"n_heads": 3},
        {"n_heads": True},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": math.nan},
    ],
)
def test_invalid_model_configuration(tokenizer, config):
    arguments = {"max_seq_len": 16, "hidden_dim": 8, "n_layers": 1, "n_heads": 2}
    arguments.update(config)
    with pytest.raises(ValueError):
        LabTOP(tokenizer, **arguments)
