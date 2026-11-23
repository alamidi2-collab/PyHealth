"""LabTOP causal language model for numeric laboratory test prediction."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


class LabTOP(nn.Module):
    """GPT-2 decoder with LabTOP's shifted next-token objective.

    Instantiate after extending the tokenizer with LabTOPProcessor. Forward
    accepts the three [batch, length] tensors returned by collate_labtop.
    """

    def __init__(self, tokenizer, max_seq_len=4096, hidden_dim=512,
                 n_layers=12, n_heads=8, dropout=0.1, lab_only=False):
        super().__init__()
        self.lab_only = lab_only
        self.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_positions=max_seq_len, n_embd=hidden_dim,
            n_layer=n_layers, n_head=n_heads, activation_function="gelu",
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ))

    def forward(self, input_ids, train_mask, position_ids):
        """Return logits [B, L, V] and mean cross entropy at supervised targets."""
        logits = self.model(input_ids=input_ids, position_ids=position_ids,
                            attention_mask=input_ids.ne(0), use_cache=False).logits
        targets = input_ids[:, 1:]
        keep = targets.ne(0)
        if self.lab_only:
            keep = keep & train_mask[:, 1:].bool()
        selected = logits[:, :-1][keep]
        loss = F.cross_entropy(selected, targets[keep]) if keep.any() else selected.sum()
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def predict_value_tokens(self, prompt_ids, end_token_id, max_new_tokens=50):
        """Greedily decode one unpadded prompt until the event delimiter."""
        device = next(self.parameters()).device
        prompt = torch.as_tensor(prompt_ids, device=device).reshape(1, -1)
        remaining = min(max_new_tokens, self.model.config.n_positions - prompt.size(1))
        if remaining <= 0:
            return []
        generated = self.model.generate(
            prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
            max_new_tokens=remaining, eos_token_id=end_token_id, pad_token_id=0,
        )[0, prompt.size(1):].tolist()
        return generated[:-1] if generated[-1] == end_token_id else generated
