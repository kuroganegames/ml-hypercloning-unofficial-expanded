"""Mixtral destination config helpers."""

from __future__ import annotations

import copy
from typing import Any

from hypercloning.common import rename_config


def build_mixtral_destination_config(
    src_config: Any,
    embedding_dim_multiplier: int,
    up_project_multiplier: int,
):
    """Return a Mixtral config expanded along supported HyperCloning axes.

    The first Mixtral implementation deliberately keeps expert count, experts
    per token, number of layers, and per-head dimension unchanged. Width is
    expanded by increasing hidden size, FFN intermediate size, and head counts.
    """

    config = copy.deepcopy(src_config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if getattr(config, "num_key_value_heads", None) is not None and config.num_key_value_heads != 1:
        config.num_key_value_heads = embedding_dim_multiplier * config.num_key_value_heads
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.tie_word_embeddings = False
    return rename_config(config, embedding_dim_multiplier, up_project_multiplier)


__all__ = ["build_mixtral_destination_config"]
