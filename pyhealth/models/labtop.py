"""LabTOP causal language model for numeric laboratory prediction.

Paper: https://proceedings.mlr.press/v287/im25a.html
The tokenizer-first constructor and existing checkpoint parameter names are
preserved. The optional dataset connects the model to PyHealth's BaseModel API.
This generative model uses its own token loss and generation evaluator.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel


class LabTOP(BaseModel):
    """GPT-2 decoder with LabTOP's shifted next-token objective.

    Instantiate after extending the tokenizer with a LabTOP processor. The
    existing loss mask supervises lab item, value, unit and event-delimiter
    tokens when lab_only is enabled, matching the released source's mask.

    Args:
        tokenizer: Extended tokenizer with padding ID zero.
        max_seq_len: Maximum context length, including generation space.
        hidden_dim: Transformer embedding width.
        n_layers: Number of decoder blocks.
        n_heads: Attention heads. Must divide hidden_dim.
        dropout: Dropout probability in [0, 1).
        lab_only: Restrict supervision to the supplied lab-token mask.
        dataset: Optional PyHealth sample dataset with raw tokens and types.

    Examples:
        >>> # tokenizer has already been extended by LabTOPProcessor.
        >>> model = LabTOP(tokenizer, 128, 32, 1, 2)  # doctest: +SKIP
        >>> result = model(input_ids, train_mask, position_ids)  # doctest: +SKIP
        >>> result["loss"].backward()  # doctest: +SKIP
    """

    def __init__(
        self,
        tokenizer: Any,
        max_seq_len: int = 4096,
        hidden_dim: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
        lab_only: bool = False,
        *,
        dataset: SampleDataset | None = None,
    ) -> None:
        super().__init__(dataset)
        sizes = (max_seq_len, hidden_dim, n_layers, n_heads)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in sizes):
            raise ValueError(
                "Context, width, layers and heads must be positive integers"
            )
        if max_seq_len < 2 or hidden_dim % n_heads:
            raise ValueError("Context must be >= 2 and hidden_dim divisible by n_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if tokenizer.pad_token_id != 0:
            raise ValueError("LabTOP collators require tokenizer padding ID zero")
        self.lab_only = lab_only
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=max_seq_len,
                n_embd=hidden_dim,
                n_layer=n_layers,
                n_head=n_heads,
                activation_function="gelu",
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )

    def _validate_batch(
        self, input_ids: Tensor, train_mask: Tensor, position_ids: Tensor
    ) -> None:
        """Reject malformed token batches before entering the transformer."""
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            raise ValueError("input_ids must be a nonempty [batch, length] tensor")
        if not 2 <= input_ids.shape[1] <= self.model.config.n_positions:
            raise ValueError(
                "Training sequence length must be between 2 and max_seq_len"
            )
        if train_mask.shape != input_ids.shape or position_ids.shape != input_ids.shape:
            raise ValueError(
                "Token IDs, masks and position IDs must have the same shape"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integers")
        if (input_ids < 0).any() or (input_ids >= self.model.config.vocab_size).any():
            raise ValueError("input_ids contain IDs outside the tokenizer vocabulary")
        if (position_ids < 0).any() or (
            position_ids >= self.model.config.n_positions
        ).any():
            raise ValueError("position_ids exceed the model context")
        if not ((train_mask == 0) | (train_mask == 1)).all():
            raise ValueError("train_mask must contain only zeros and ones")

    def _loss(self, logits: Tensor, input_ids: Tensor, train_mask: Tensor) -> Tensor:
        """Return token-weighted cross entropy, or differentiable zero if empty."""
        targets = input_ids[:, 1:]
        keep = targets.ne(0)
        if self.lab_only:
            keep = keep & train_mask[:, 1:].bool()
        selected = logits[:, :-1][keep]
        return (
            F.cross_entropy(selected, targets[keep]) if keep.any() else selected.sum()
        )

    def forward(
        self, input_ids: Tensor, train_mask: Tensor, position_ids: Tensor
    ) -> dict[str, Tensor]:
        """Compute a scalar loss and logits shaped [batch, length, vocabulary].

        Args:
            input_ids: Right-padded integer token IDs.
            train_mask: Aligned binary mask for lab-token supervision.
            position_ids: Aligned integer positions, padded with zeros.

        Returns:
            Loss and logits for this autoregressive objective. Numeric lab
            predictions require predict_value_tokens and the LabTOP evaluator.
        """
        self._validate_batch(input_ids, train_mask, position_ids)
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=input_ids.ne(0),
            use_cache=False,
        ).logits
        return {"loss": self._loss(logits, input_ids, train_mask), "logits": logits}

    def forward_from_embedding(
        self,
        embeddings: Tensor,
        input_ids: Tensor,
        train_mask: Tensor,
        position_ids: Tensor,
    ) -> dict[str, Tensor]:
        """Run the same objective from embeddings for gradient-based inspection.

        Args:
            embeddings: Token embeddings shaped [batch, length, hidden_dim].
            input_ids: Original token IDs, used as targets and padding mask.
            train_mask: Binary mask selecting supervised lab targets.
            position_ids: Token positions with the same shape as input_ids.

        Returns:
            The same loss/logits dictionary as forward.
        """
        self._validate_batch(input_ids, train_mask, position_ids)
        expected = (*input_ids.shape, self.model.config.n_embd)
        if tuple(embeddings.shape) != expected or not torch.isfinite(embeddings).all():
            raise ValueError("embeddings must have finite values and shape [B, L, D]")
        logits = self.model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            attention_mask=input_ids.ne(0),
            use_cache=False,
        ).logits
        return {"loss": self._loss(logits, input_ids, train_mask), "logits": logits}

    @torch.no_grad()
    def predict_value_tokens(
        self,
        prompt_ids: Tensor | list[int],
        end_token_id: int,
        max_new_tokens: int = 50,
    ) -> list[int]:
        """Greedily decode an unpadded prompt and restore all training modes.

        Args:
            prompt_ids: One nonempty token sequence containing no padding.
            end_token_id: Event delimiter at which generation stops.
            max_new_tokens: Positive generation budget. The prompt builder must
                reserve at least this much context when comparing budgets.

        Returns:
            Generated token IDs without the final event delimiter.

        Raises:
            ValueError: For an invalid prompt, delimiter, or generation budget.
        """
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        prompt = torch.as_tensor(prompt_ids, device=self.device)
        if prompt.ndim != 1 or prompt.numel() == 0:
            raise ValueError("prompt_ids must be one nonempty, unpadded sequence")
        if prompt.dtype not in (torch.int32, torch.int64):
            raise ValueError("prompt_ids must contain integer token IDs")
        if (prompt <= 0).any() or (prompt >= self.model.config.vocab_size).any():
            raise ValueError("prompt contains padding or IDs outside the vocabulary")
        if (
            isinstance(end_token_id, bool)
            or not isinstance(end_token_id, int)
            or not 0 < end_token_id < len(self.model.get_input_embeddings().weight)
        ):
            raise ValueError(
                "end_token_id must identify a non-padding vocabulary token"
            )
        remaining = min(max_new_tokens, self.model.config.n_positions - prompt.numel())
        if remaining < 1:
            raise ValueError("Prompt leaves no context space for generation")
        states = {module: module.training for module in self.modules()}
        try:
            self.eval()
            batch = prompt.reshape(1, -1)
            generated = self.model.generate(
                batch,
                attention_mask=torch.ones_like(batch),
                do_sample=False,
                max_new_tokens=remaining,
                eos_token_id=end_token_id,
                pad_token_id=0,
            )[0, batch.size(1) :].tolist()
        finally:
            for module, state in states.items():
                module.training = state
        if generated and generated[-1] == end_token_id:
            generated.pop()
        return generated
