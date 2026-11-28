"""LabTOP causal language model for numeric laboratory test prediction."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2LMHeadModel


class LabTOP(nn.Module):
    """GPT-2 decoder with LabTOP's shifted next-token objective.

    Instantiate after extending the tokenizer with LabTOPHiRIDProcessor. Forward
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
    def predict_value_tokens(self, prompt_ids, end_token_id, max_new_tokens=50, *, return_entropies=False):
        """Decode a prompt, optionally returning full-vocabulary entropy per generated token (nats)."""
        device = next(self.parameters()).device
        prompt = torch.as_tensor(prompt_ids, device=device).reshape(1, -1)
        remaining = min(max_new_tokens, self.model.config.n_positions - prompt.size(1))
        if remaining <= 0:
            return ([], []) if return_entropies else []
        output = self.model.generate(
            prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
            max_new_tokens=remaining, eos_token_id=end_token_id, pad_token_id=0,
            return_dict_in_generate=return_entropies, output_logits=return_entropies,
        )
        sequences = output.sequences if return_entropies else output
        ids = sequences[0, prompt.size(1):].tolist()
        if ids and ids[-1] == end_token_id:
            ids.pop()
        if not return_entropies:
            return ids
        if not ids:
            return [], []
        logits = torch.cat(output.logits[:len(ids)]).float()
        entropies = torch.special.entr(logits.softmax(dim=-1)).sum(dim=-1)
        return ids, entropies.tolist()
