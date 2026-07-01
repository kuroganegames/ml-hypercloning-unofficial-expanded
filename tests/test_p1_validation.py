from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from hc_validate import ValidationSpec
from hc_validate.generation import GenerationCacheSpec, run_generation_cache_check
from hc_validate.inputs import InputMatrixSpec
from hc_validate.long_context import LongContextSpec, infer_context_lengths, run_long_context_check
from hc_validate.p1_suite import P1ValidationSpec, run_p1_validation
from hypercloning.common import clone_matrix


class ToyCacheLM(nn.Module):
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
            torch.manual_seed(1234)
            self.embed_tokens.weight.data.normal_()
            self.embed_positions.weight.data.normal_()
            self.lm_head.weight.data.normal_()
        else:
            self.embed_tokens.weight.data = clone_matrix(self.embed_tokens.weight.shape, base.embed_tokens.weight.data, normalize=False)
            self.embed_positions.weight.data = clone_matrix(self.embed_positions.weight.shape, base.embed_positions.weight.data, normalize=False)
            self.lm_head.weight.data = clone_matrix(self.lm_head.weight.shape, base.lm_head.weight.data, normalize=True)

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
            past_len = int(past_key_values.get("past_len", 0)) if past_key_values is not None else input_ids.shape[1]
            if past_key_values is not None:
                past_len += input_ids.shape[1]
            new_past = {"past_len": past_len}
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
    source = ToyCacheLM()
    destination = ToyCacheLM(hidden_size=8, intermediate_size=16, num_heads=4, base=source)
    return source, destination


def make_batch():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long)
    mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}


def base_spec():
    return ValidationSpec(
        seq_len=8,
        batch_size=2,
        num_batches=1,
        embedding_dim_multiplier=2,
        up_project_multiplier=2,
        num_heads_multiplier=2,
        device="cpu",
        compare_hidden_states=False,
    )


def test_generation_cache_check_passes_with_toy_cache_model():
    source, destination = make_pair()
    report = run_generation_cache_check(
        source,
        destination,
        [make_batch()],
        base_spec(),
        GenerationCacheSpec(max_new_tokens=3, num_prompt_batches=1, require_past_key_values=True),
    )
    assert report.ok, report.failures


def test_long_context_check_passes_for_position_and_mask_patterns():
    source, destination = make_pair()
    assert infer_context_lengths(source.config, LongContextSpec(lengths=(4, 8), include_inferred_lengths=False)) == (4, 8)
    report = run_long_context_check(
        source,
        destination,
        tokenizer=None,
        base_spec=base_spec(),
        long_spec=LongContextSpec(
            lengths=(4, 8, 16),
            include_inferred_lengths=False,
            patterns=("no_padding", "right_padding", "left_padding", "ragged", "same_token", "alternating", "explicit_position_ids"),
            run_generation_cache_at_long_context=False,
        ),
    )
    assert report.ok, report.failures


def test_p1_suite_runs_generation_and_long_context():
    source, destination = make_pair()
    report = run_p1_validation(
        source,
        destination,
        tokenizer=None,
        spec=P1ValidationSpec(
            base=base_spec(),
            prompt_inputs=InputMatrixSpec(seq_lengths=(8,), tokenizer_fixture_batches=False, chat_template_batches=False, num_random_batches_per_length=1),
            generation=GenerationCacheSpec(max_new_tokens=2, num_prompt_batches=1),
            long_context=LongContextSpec(lengths=(8,), include_inferred_lengths=False, run_generation_cache_at_long_context=False),
        ),
    )
    assert report.ok, report.failures
