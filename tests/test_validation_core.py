from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from hc_validate import (
    ValidationSpec,
    audit_config,
    audit_state_with_trace,
    make_random_lm_batches,
    run_train_smoke,
    validate_forward,
)
from hypercloning.common import clone_matrix
from hypercloning.trace import CloneTrace


class ToyLM(nn.Module):
    def __init__(self, vocab_size=17, hidden_size=4, intermediate_size=8, num_heads=2, repeat=1, base=None):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_hidden_layers=1,
        )
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if base is not None:
            self.embed_tokens.weight.data = clone_matrix(self.embed_tokens.weight.shape, base.embed_tokens.weight.data, normalize=False)
            self.lm_head.weight.data = clone_matrix(self.lm_head.weight.shape, base.lm_head.weight.data, normalize=True)
        elif repeat == 1:
            torch.manual_seed(0)
            self.embed_tokens.weight.data.normal_()
            self.lm_head.weight.data.normal_()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
        output_attentions=False,
    ):
        hidden = self.embed_tokens(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        hidden_states = (hidden,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden_states, attentions=None)


def test_forward_config_state_and_train_smoke():
    source = ToyLM()
    destination = ToyLM(hidden_size=8, intermediate_size=16, num_heads=4, base=source)
    spec = ValidationSpec(seq_len=8, batch_size=2, num_batches=3, embedding_dim_multiplier=2, up_project_multiplier=2, num_heads_multiplier=2)
    batches = make_random_lm_batches(source.config.vocab_size, spec)

    config_report = audit_config(source.config, destination.config, spec)
    assert config_report.ok, config_report.failures

    forward_report = validate_forward(source, destination, batches, spec)
    assert forward_report.ok, forward_report.failures

    trace = CloneTrace()
    trace.add_matrix('embed_tokens.weight', 'embed_tokens.weight', tuple(source.embed_tokens.weight.shape), tuple(destination.embed_tokens.weight.shape), kind='embedding', normalize_input_repeat=False)
    trace.add_matrix('lm_head.weight', 'lm_head.weight', tuple(source.lm_head.weight.shape), tuple(destination.lm_head.weight.shape), kind='linear', normalize_input_repeat=True)
    state_report = audit_state_with_trace(source, destination, trace)
    assert state_report.ok, state_report.failures

    smoke_report = run_train_smoke(destination, batches, steps=2)
    assert smoke_report.ok, smoke_report.failures
