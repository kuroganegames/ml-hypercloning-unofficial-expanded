"""HyperCloning implementation for Hugging Face Mixtral models."""

from __future__ import annotations

import torch
from transformers import MixtralForCausalLM

from hypercloning.common import clone_linear_layer, clone_matrix, clone_rms_norm, clone_vector
from hypercloning.gemma_cloning import clone_gemma_qkv_layer
from hypercloning.models.mixtral.config import build_mixtral_destination_config
from hypercloning.models.mixtral.layout import (
    LEGACY_MODULELIST_EXPERTS,
    PACKED_EXPERTS,
    detect_mixtral_moe_layout,
    get_mixtral_moe,
)


def clone_mixtral_attention(dst_layer, src_layer, snr_db=None):
    """Clone Mixtral attention while keeping head_dim unchanged."""

    clone_gemma_qkv_layer(
        dst_layer.q_proj,
        src_layer.q_proj,
        dst_layer.config.num_attention_heads,
        src_layer.config.num_attention_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.k_proj,
        src_layer.k_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.v_proj,
        src_layer.v_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)
    return dst_layer


def _clone_router(dst_router, src_router) -> None:
    """Clone MoE router weights without changing expert count.

    Router top-k selection is discontinuous, so noise is intentionally not added
    to router weights even if ``snr_db`` is used for dense/expert layers.
    """

    dst_router.weight.data = clone_matrix(
        dst_router.weight.shape,
        src_router.weight.data,
        snr_db=None,
        normalize=True,
    )
    if getattr(src_router, "bias", None) is not None:
        assert getattr(dst_router, "bias", None) is not None, "source router has bias but destination router does not"
        dst_router.bias.data = clone_vector(dst_router.bias.shape, src_router.bias.data)


def _clone_packed_gate_up(dst_gate_up: torch.nn.Parameter, src_gate_up: torch.nn.Parameter, snr_db=None) -> None:
    """Clone packed expert gate/up tensors.

    Packed Mixtral experts store gate and up projections as
    ``[num_experts, 2 * intermediate_size, hidden_size]``. Split/repack avoids
    accidentally interleaving gate and up blocks after FFN expansion.
    """

    if dst_gate_up.ndim != 3 or src_gate_up.ndim != 3:
        raise AssertionError("packed Mixtral gate_up_proj must be a 3D parameter")
    if dst_gate_up.shape[0] != src_gate_up.shape[0]:
        raise AssertionError("HyperCloning for Mixtral keeps num_local_experts unchanged")
    if dst_gate_up.shape[1] % 2 != 0 or src_gate_up.shape[1] % 2 != 0:
        raise AssertionError("Mixtral packed gate_up_proj must have an even gate/up dimension")

    src_gate, src_up = src_gate_up.data.chunk(2, dim=1)
    dst_intermediate = dst_gate_up.shape[1] // 2
    cloned = torch.empty_like(dst_gate_up.data)

    for expert_idx in range(src_gate_up.shape[0]):
        cloned[expert_idx, :dst_intermediate, :] = clone_matrix(
            (dst_intermediate, dst_gate_up.shape[2]),
            src_gate[expert_idx],
            snr_db=snr_db,
            normalize=True,
        )
        cloned[expert_idx, dst_intermediate:, :] = clone_matrix(
            (dst_intermediate, dst_gate_up.shape[2]),
            src_up[expert_idx],
            snr_db=snr_db,
            normalize=True,
        )

    dst_gate_up.data = cloned


def _clone_packed_down(dst_down: torch.nn.Parameter, src_down: torch.nn.Parameter, snr_db=None) -> None:
    """Clone packed expert down-projection tensors."""

    if dst_down.ndim != 3 or src_down.ndim != 3:
        raise AssertionError("packed Mixtral down_proj must be a 3D parameter")
    if dst_down.shape[0] != src_down.shape[0]:
        raise AssertionError("HyperCloning for Mixtral keeps num_local_experts unchanged")

    cloned = torch.empty_like(dst_down.data)
    for expert_idx in range(src_down.shape[0]):
        cloned[expert_idx] = clone_matrix(
            dst_down.shape[1:],
            src_down.data[expert_idx],
            snr_db=snr_db,
            normalize=True,
        )
    dst_down.data = cloned


def _clone_packed_moe(dst_moe, src_moe, snr_db=None) -> None:
    _clone_router(dst_moe.gate, src_moe.gate)
    _clone_packed_gate_up(dst_moe.experts.gate_up_proj, src_moe.experts.gate_up_proj, snr_db=snr_db)
    _clone_packed_down(dst_moe.experts.down_proj, src_moe.experts.down_proj, snr_db=snr_db)


def _clone_legacy_modulelist_moe(dst_moe, src_moe, snr_db=None) -> None:
    _clone_router(dst_moe.gate, src_moe.gate)
    if len(dst_moe.experts) != len(src_moe.experts):
        raise AssertionError("HyperCloning for Mixtral keeps num_local_experts unchanged")

    for dst_expert, src_expert in zip(dst_moe.experts, src_moe.experts):
        # w1 = gate projection, w3 = up projection, w2 = down projection.
        clone_linear_layer(dst_expert.w1, src_expert.w1, snr_db=snr_db)
        clone_linear_layer(dst_expert.w3, src_expert.w3, snr_db=snr_db)
        clone_linear_layer(dst_expert.w2, src_expert.w2, snr_db=snr_db)


def clone_mixtral_moe(dst_moe, src_moe, snr_db=None):
    """Clone a Mixtral MoE block across supported Transformers layouts."""

    dst_layout = detect_mixtral_moe_layout(dst_moe)
    src_layout = detect_mixtral_moe_layout(src_moe)
    if dst_layout != src_layout:
        raise AssertionError(f"Mixtral source/destination MoE layout mismatch: source={src_layout}, destination={dst_layout}")

    if dst_layout == PACKED_EXPERTS:
        _clone_packed_moe(dst_moe, src_moe, snr_db=snr_db)
        return dst_moe

    if dst_layout == LEGACY_MODULELIST_EXPERTS:
        _clone_legacy_modulelist_moe(dst_moe, src_moe, snr_db=snr_db)
        return dst_moe

    raise AssertionError(
        "Unsupported Mixtral MoE layout. Expected packed experts.gate_up_proj/down_proj "
        "or legacy ModuleList experts with w1/w2/w3."
    )


def clone_mixtral(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """Clone a Hugging Face MixtralForCausalLM model.

    Supported expansion axes are hidden size, FFN intermediate size, attention
    head count, and non-MQA KV head count. Expert count and top-k routing are
    intentionally fixed in this first implementation.
    """

    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert (
        num_heads_multiplier == embedding_dim_multiplier
    ), "head_dim expansion is not supported for Mixtral; expand the number of heads instead"

    config = build_mixtral_destination_config(src_network.config, embedding_dim_multiplier, up_project_multiplier)
    dst_network = MixtralForCausalLM._from_config(config)

    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm)
        dst_layer.self_attn = clone_mixtral_attention(dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db)
        clone_mixtral_moe(get_mixtral_moe(dst_layer), get_mixtral_moe(src_layer), snr_db=snr_db)

    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network


__all__ = ["clone_mixtral", "clone_mixtral_attention", "clone_mixtral_moe"]
