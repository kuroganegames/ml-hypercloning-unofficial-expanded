"""Destination configuration helpers for dense-only AFMoE HyperCloning."""

from __future__ import annotations

import copy
from numbers import Integral
from typing import Any

from hypercloning.common import rename_config


def _validate_multiplier(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def build_afmoe_dense_destination_config(
    src_config: Any,
    embedding_dim_multiplier: int,
    up_project_multiplier: int,
):
    """Return an AFMoE config expanded along function-preserving width axes.

    The dense-only implementation keeps layer count, attention head dimension,
    hybrid attention layout, context settings, and all MoE-related configuration
    values unchanged. Hidden width is expanded by increasing the number of
    attention heads rather than ``head_dim``. MQA keeps one KV head; GQA/MHA KV
    head counts expand with hidden width.

    Destination input embeddings and the LM head require different scaling, so
    the destination is deliberately configured with untied word embeddings.
    """

    embedding_dim_multiplier = _validate_multiplier(
        "embedding_dim_multiplier", embedding_dim_multiplier
    )
    up_project_multiplier = _validate_multiplier(
        "up_project_multiplier", up_project_multiplier
    )

    config = copy.deepcopy(src_config)

    config.hidden_size = embedding_dim_multiplier * int(config.hidden_size)
    config.intermediate_size = up_project_multiplier * int(config.intermediate_size)
    config.num_attention_heads = embedding_dim_multiplier * int(config.num_attention_heads)

    num_key_value_heads = int(config.num_key_value_heads)
    if num_key_value_heads != 1:
        config.num_key_value_heads = embedding_dim_multiplier * num_key_value_heads

    # Keep an explicit list rather than relying on AfmoeConfig.__post_init__ to
    # regenerate the source model's full/sliding attention pattern.
    config.layer_types = list(src_config.layer_types)

    # A tied destination cannot simultaneously preserve the unscaled input
    # embedding and the 1 / hidden-repeat scaled output projection.
    config.tie_word_embeddings = False

    if getattr(config, "_name_or_path", None) is None:
        config._name_or_path = ""
    return rename_config(config, embedding_dim_multiplier, up_project_multiplier)


__all__ = ["build_afmoe_dense_destination_config"]
