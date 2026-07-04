from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch


transformers = pytest.importorskip("transformers")


def llama_classes():
    """Return Llama classes when the installed Transformers version exposes them."""

    LlamaConfig = getattr(transformers, "LlamaConfig", None)
    LlamaForCausalLM = getattr(transformers, "LlamaForCausalLM", None)
    if LlamaConfig is None or LlamaForCausalLM is None:
        pytest.skip("transformers does not expose LlamaConfig/LlamaForCausalLM")
    return LlamaConfig, LlamaForCausalLM


def _declares_init_arg(cls: type[Any], name: str) -> bool:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _accepts_init_arg(cls: type[Any], name: str) -> bool:
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def make_tiny_llama3_like_config(
    *,
    num_key_value_heads: int = 2,
    explicit_head_dim: bool = False,
    **overrides: Any,
):
    """Build a tiny Llama3-like config without downloading gated checkpoints."""

    LlamaConfig, _ = llama_classes()
    hidden_size = int(overrides.pop("hidden_size", 32))
    num_attention_heads = int(overrides.pop("num_attention_heads", 4))

    kwargs: dict[str, Any] = {
        "vocab_size": 128,
        "hidden_size": hidden_size,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "max_position_embeddings": 256,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }

    if _accepts_init_arg(LlamaConfig, "attention_dropout"):
        kwargs["attention_dropout"] = 0.0
    if _accepts_init_arg(LlamaConfig, "attention_bias"):
        kwargs["attention_bias"] = False
    if _accepts_init_arg(LlamaConfig, "mlp_bias"):
        kwargs["mlp_bias"] = False

    # Only pass head_dim when the installed config explicitly supports it. Older
    # Transformers versions may accept arbitrary **kwargs but ignore head_dim in
    # modeling code, which would make this test less meaningful.
    if explicit_head_dim and _declares_init_arg(LlamaConfig, "head_dim"):
        kwargs["head_dim"] = hidden_size // num_attention_heads

    kwargs.update(overrides)
    return LlamaConfig(**kwargs)


def force_eager(model_or_config: Any) -> None:
    """Prefer the deterministic eager attention path when Transformers supports it."""

    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"


def make_random_inputs(config: Any, *, batch_size: int = 2, seq_len: int = 16) -> dict[str, torch.Tensor]:
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
