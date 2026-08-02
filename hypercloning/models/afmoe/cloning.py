"""Function-preserving HyperCloning for dense-only Hugging Face AFMoE models."""

from __future__ import annotations

import copy

from transformers import AfmoeForCausalLM

from hypercloning.common import clone_linear_layer, clone_matrix, clone_rms_norm
from hypercloning.gemma_cloning import clone_gemma_qkv_layer
from hypercloning.models.afmoe.config import build_afmoe_dense_destination_config
from hypercloning.models.afmoe.guards import assert_supported_dense_afmoe


def clone_afmoe_rms_norm(dst_norm, src_norm):
    """Clone AFMoE RMSNorm weights after checking its epsilon invariant."""

    dst_eps = getattr(dst_norm, "variance_epsilon", None)
    src_eps = getattr(src_norm, "variance_epsilon", None)
    if dst_eps != src_eps:
        raise AssertionError(
            f"AFMoE RMSNorm epsilon mismatch: source={src_eps!r}, destination={dst_eps!r}"
        )
    clone_rms_norm(dst_norm, src_norm)
    return dst_norm


def clone_afmoe_attention(dst_attention, src_attention, snr_db=None):
    """Clone AFMoE Q/K/V, per-head norms, sigmoid gate, and output projection."""

    src_head_dim = int(src_attention.head_dim)
    dst_head_dim = int(dst_attention.head_dim)
    if dst_head_dim != src_head_dim:
        raise AssertionError(
            "AFMoE head_dim expansion is not supported; "
            f"source={src_head_dim}, destination={dst_head_dim}"
        )

    src_config = src_attention.config
    dst_config = dst_attention.config

    clone_gemma_qkv_layer(
        dst_attention.q_proj,
        src_attention.q_proj,
        dst_config.num_attention_heads,
        src_config.num_attention_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_attention.k_proj,
        src_attention.k_proj,
        dst_config.num_key_value_heads,
        src_config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_attention.v_proj,
        src_attention.v_proj,
        dst_config.num_key_value_heads,
        src_config.num_key_value_heads,
        snr_db=snr_db,
    )

    # The attention gate is laid out per query head and must use exactly the
    # same head replication order as q_proj.
    clone_gemma_qkv_layer(
        dst_attention.gate_proj,
        src_attention.gate_proj,
        dst_config.num_attention_heads,
        src_config.num_attention_heads,
        snr_db=snr_db,
    )

    clone_afmoe_rms_norm(dst_attention.q_norm, src_attention.q_norm)
    clone_afmoe_rms_norm(dst_attention.k_norm, src_attention.k_norm)
    clone_linear_layer(dst_attention.o_proj, src_attention.o_proj, snr_db=snr_db)
    return dst_attention


def clone_afmoe(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """HyperClone a Hugging Face ``AfmoeForCausalLM`` with all-dense MLPs.

    Supported axes are hidden size, dense FFN intermediate size, query-head
    count, and non-MQA KV-head count. Layer count, head dimension, hybrid
    attention pattern, and all MoE settings remain unchanged. Source models
    containing any routed/shared expert layer are rejected.
    """

    assert_supported_dense_afmoe(src_network)

    snr_db = kwargs.get("snr_db", None)
    if snr_db is not None:
        raise NotImplementedError(
            "dense-only AFMoE PR 1 supports exact cloning only; "
            "snr_db noise is not implemented"
        )

    num_heads_multiplier = kwargs.get(
        "num_heads_multiplier", embedding_dim_multiplier
    )
    if num_heads_multiplier != embedding_dim_multiplier:
        raise AssertionError(
            "AFMoE head_dim expansion is not supported; "
            "num_heads_multiplier must equal embedding_dim_multiplier"
        )

    config = build_afmoe_dense_destination_config(
        src_network.config,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
    )

    attn_implementation = kwargs.get("attn_implementation", None)
    if attn_implementation is not None and hasattr(config, "_attn_implementation"):
        config._attn_implementation = attn_implementation

    dst_network = AfmoeForCausalLM._from_config(config)

    # Input embedding must produce repeated hidden states without scaling.
    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    for dst_layer, src_layer in zip(
        dst_network.model.layers,
        src_network.model.layers,
        strict=True,
    ):
        for norm_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
        ):
            clone_afmoe_rms_norm(
                getattr(dst_layer, norm_name),
                getattr(src_layer, norm_name),
            )

        clone_afmoe_attention(
            dst_layer.self_attn,
            src_layer.self_attn,
            snr_db=snr_db,
        )

        clone_linear_layer(
            dst_layer.mlp.gate_proj,
            src_layer.mlp.gate_proj,
            snr_db=snr_db,
        )
        clone_linear_layer(
            dst_layer.mlp.up_proj,
            src_layer.mlp.up_proj,
            snr_db=snr_db,
        )
        clone_linear_layer(
            dst_layer.mlp.down_proj,
            src_layer.mlp.down_proj,
            snr_db=snr_db,
        )

    clone_afmoe_rms_norm(dst_network.model.norm, src_network.model.norm)

    # Destination embeddings are intentionally untied. clone_linear_layer
    # divides the repeated input dimension by embedding_dim_multiplier, which
    # preserves logits for repeated final hidden states.
    clone_linear_layer(dst_network.lm_head, src_network.lm_head, snr_db=None)

    if dst_network.model.embed_tokens.weight is dst_network.lm_head.weight:
        raise AssertionError("AFMoE destination input and output embeddings must be untied")

    if hasattr(src_network, "generation_config"):
        dst_network.generation_config = copy.deepcopy(src_network.generation_config)

    parameters = list(src_network.parameters())
    if parameters:
        dst_network.to(device=parameters[0].device)
    dst_network.train(src_network.training)
    return dst_network


__all__ = [
    "clone_afmoe",
    "clone_afmoe_attention",
    "clone_afmoe_rms_norm",
]
