from __future__ import annotations

from typing import Any

import pytest
import torch


transformers = pytest.importorskip("transformers")


def afmoe_classes():
    """Return AFMoE classes when the installed Transformers exposes them."""

    AfmoeConfig = getattr(transformers, "AfmoeConfig", None)
    AfmoeForCausalLM = getattr(transformers, "AfmoeForCausalLM", None)
    if AfmoeConfig is None or AfmoeForCausalLM is None:
        pytest.skip("transformers does not expose AfmoeConfig/AfmoeForCausalLM")
    return AfmoeConfig, AfmoeForCausalLM


def make_tiny_dense_afmoe_config(
    *,
    num_key_value_heads: int = 2,
    num_dense_layers: int = 2,
    mup_enabled: bool = False,
    tie_word_embeddings: bool = True,
    **overrides: Any,
):
    """Build a tiny all-dense AFMoE config matching the notebook invariants."""

    AfmoeConfig, _ = afmoe_classes()
    kwargs: dict[str, Any] = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_dense_layers": num_dense_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 64,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": False,
        "tie_word_embeddings": tie_word_embeddings,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "route_scale": 1.0,
        "output_router_logits": False,
        "global_attn_every_n_layers": 2,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "attention_dropout": 0.0,
        "attention_bias": False,
        "mup_enabled": mup_enabled,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    kwargs.update(overrides)
    return AfmoeConfig(**kwargs)


def force_eager(model_or_config: Any) -> None:
    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"


def make_random_inputs(config: Any, *, batch_size: int = 2, seq_len: int = 12):
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
