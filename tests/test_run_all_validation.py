from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from hc_validate.run_all import build_all_validation_spec, infer_validation_spec, run_all_validations
from hypercloning.common import clone_matrix


class ToyAllValidationLM(nn.Module):
    def __init__(self, vocab_size=23, hidden_size=4, intermediate_size=8, num_heads=2, max_position_embeddings=32, base=None):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=1,
            max_position_embeddings=max_position_embeddings,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if base is None:
            torch.manual_seed(2024)
            self.embed_tokens.weight.data.normal_()
            self.embed_positions.weight.data.normal_()
            self.lm_head.weight.data.normal_()
        else:
            self.embed_tokens.weight.data = clone_matrix(self.embed_tokens.weight.shape, base.embed_tokens.weight.data, normalize=False)
            self.embed_positions.weight.data = clone_matrix(self.embed_positions.weight.shape, base.embed_positions.weight.data, normalize=False)
            self.lm_head.weight.data = clone_matrix(self.lm_head.weight.shape, base.lm_head.weight.data, normalize=True)

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"config": dict(self.config.__dict__), "state_dict": self.state_dict()}, path / "toy_model.pt")

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        data = torch.load(Path(path) / "toy_model.pt", map_location="cpu")
        cfg = data["config"]
        model = cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_heads=cfg["num_attention_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
        )
        model.load_state_dict(data["state_dict"])
        return model

    def _position_ids(self, input_ids, attention_mask=None, past_key_values=None, position_ids=None):
        if position_ids is not None:
            return position_ids.to(input_ids.device)
        if past_key_values is not None:
            start = int(past_key_values.get("past_len", 0))
            row = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
            return row.unsqueeze(0).expand(input_ids.shape[0], -1)
        if attention_mask is not None:
            pos = attention_mask.long().cumsum(dim=-1) - 1
            return pos.masked_fill(attention_mask == 0, 0)
        return torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
        output_attentions=False,
        position_ids=None,
    ):
        position_ids = self._position_ids(input_ids, attention_mask, past_key_values, position_ids)
        position_ids = position_ids.clamp_max(self.config.max_position_embeddings - 1)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None and logits.shape[1] > 1:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        hidden_states = (hidden,) if output_hidden_states else None
        new_past = None
        if use_cache:
            new_past = {"past_len": int(past_key_values.get("past_len", 0)) + input_ids.shape[1]} if past_key_values is not None else {"past_len": input_ids.shape[1]}
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states, attentions=None, past_key_values=new_past)

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=2, return_dict_in_generate=True, output_scores=False, output_logits=False, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        sequences = input_ids.clone()
        mask = attention_mask.clone()
        past = None
        next_token = None
        logits_steps = []
        for step in range(max_new_tokens):
            if step == 0:
                out = self.forward(sequences, attention_mask=mask, use_cache=True, return_dict=True)
                lengths = mask.long().sum(dim=-1).clamp_min(1) - 1
                batch_idx = torch.arange(sequences.shape[0], device=sequences.device)
                logits = out.logits[batch_idx, lengths, :]
            else:
                out = self.forward(next_token, attention_mask=mask, past_key_values=past, use_cache=True, return_dict=True)
                logits = out.logits[:, -1, :]
            past = out.past_key_values
            logits_steps.append(logits)
            next_token = logits.argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=-1)
        if return_dict_in_generate:
            return SimpleNamespace(sequences=sequences, logits=tuple(logits_steps), scores=tuple(logits_steps))
        return sequences


def make_pair():
    source = ToyAllValidationLM()
    destination = ToyAllValidationLM(hidden_size=8, intermediate_size=16, num_heads=4, base=source)
    return source, destination


def test_infer_validation_spec_from_configs():
    source, destination = make_pair()
    spec, warnings = infer_validation_spec(source.config, destination.config, device="cpu", seq_len=4, batch_size=2, num_batches=1)
    assert warnings == []
    assert spec.embedding_dim_multiplier == 2
    assert spec.up_project_multiplier == 2
    assert spec.num_heads_multiplier == 2


def test_run_all_validations_writes_individual_reports_and_summary(tmp_path):
    source, destination = make_pair()
    base_spec, warnings = infer_validation_spec(source.config, destination.config, device="cpu", seq_len=4, batch_size=2, num_batches=1)
    assert warnings == []
    spec = build_all_validation_spec(
        base_spec,
        output_dir=tmp_path,
        p0_seq_lengths=(4,),
        p1_prompt_seq_lengths=(4,),
        long_context_lengths=(4,),
        long_context_max_length_cap=4,
        max_new_tokens=2,
        run_save_reload=True,
        run_generate=True,
        run_manual_cache=True,
        run_long_context=True,
        metadata={"test": True},
    )
    spec.p0.save_reload.allow_transformers_fallback = False
    spec.p1.long_context.run_generation_cache_at_long_context = False
    result = run_all_validations(source, destination, tokenizer=None, spec=spec)
    assert result.summary["ok"], result.summary
    assert (tmp_path / "p0_report.json").exists()
    assert (tmp_path / "p1_report.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "all_reports.json").exists()
