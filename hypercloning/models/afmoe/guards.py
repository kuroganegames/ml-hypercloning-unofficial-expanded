"""Strict support checks for the first dense-only AFMoE implementation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class UnsupportedAfmoeConfigurationError(ValueError):
    """Raised when an AFMoE model is outside the dense-only PR-1 scope."""


_FORBIDDEN_MOE_PARAMETER_TOKENS = (
    ".router.",
    ".experts.",
    ".shared_experts.",
)


def _parameter_devices(parameters: Iterable[nn.Parameter]) -> set[torch.device]:
    return {parameter.device for parameter in parameters}


def assert_supported_dense_afmoe(model) -> None:
    """Validate that ``model`` is a materialized, single-device, all-dense AFMoE LM.

    PR 1 intentionally rejects routed/shared experts, muP input scaling,
    quantized projection replacements, and sharded/meta-device models. Those
    paths need separate cloning and loading designs rather than silent partial
    support.
    """

    config = getattr(model, "config", None)
    if config is None:
        raise UnsupportedAfmoeConfigurationError("AFMoE source model must expose .config")

    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    num_dense_layers = getattr(config, "num_dense_layers", None)
    if num_dense_layers != num_hidden_layers:
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE HyperCloning requires "
            "num_dense_layers == num_hidden_layers; "
            f"got {num_dense_layers!r} and {num_hidden_layers!r}"
        )

    if bool(getattr(config, "mup_enabled", False)):
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE PR 1 does not support mup_enabled=True"
        )

    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layer_types, (list, tuple)) or len(layer_types) != num_hidden_layers:
        raise UnsupportedAfmoeConfigurationError(
            "AFMoE layer_types must contain one entry per hidden layer"
        )
    unsupported_layer_types = sorted(
        set(layer_types).difference({"full_attention", "sliding_attention"})
    )
    if unsupported_layer_types:
        raise UnsupportedAfmoeConfigurationError(
            f"unsupported AFMoE layer_types: {unsupported_layer_types}"
        )

    num_attention_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_key_value_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    if num_attention_heads < 1 or num_key_value_heads < 1:
        raise UnsupportedAfmoeConfigurationError(
            "AFMoE attention and KV head counts must be positive"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise UnsupportedAfmoeConfigurationError(
            "AFMoE num_attention_heads must be divisible by num_key_value_heads"
        )

    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None and int(head_dim) < 1:
        raise UnsupportedAfmoeConfigurationError("AFMoE head_dim must be positive")

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != num_hidden_layers:
        raise UnsupportedAfmoeConfigurationError(
            "AFMoE source model must expose model.layers matching num_hidden_layers"
        )

    if not isinstance(getattr(backbone, "embed_tokens", None), nn.Embedding):
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE PR 1 requires a standard nn.Embedding input layer"
        )
    if not isinstance(getattr(model, "lm_head", None), nn.Linear):
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE PR 1 requires a standard nn.Linear lm_head"
        )

    for layer_index, layer in enumerate(layers):
        if getattr(layer, "moe_enabled", None) is not False:
            raise UnsupportedAfmoeConfigurationError(
                f"AFMoE layer {layer_index} is not a dense MLP layer"
            )

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise UnsupportedAfmoeConfigurationError(
                f"AFMoE layer {layer_index} does not expose .mlp"
            )
        for name in ("router", "experts", "shared_experts"):
            if hasattr(mlp, name):
                raise UnsupportedAfmoeConfigurationError(
                    f"AFMoE layer {layer_index} unexpectedly exposes MoE module {name!r}"
                )
        for name in ("gate_proj", "up_proj", "down_proj"):
            if not isinstance(getattr(mlp, name, None), nn.Linear):
                raise UnsupportedAfmoeConfigurationError(
                    f"AFMoE layer {layer_index} requires standard nn.Linear mlp.{name}"
                )

        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise UnsupportedAfmoeConfigurationError(
                f"AFMoE layer {layer_index} does not expose .self_attn"
            )
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            if not isinstance(getattr(attention, name, None), nn.Linear):
                raise UnsupportedAfmoeConfigurationError(
                    f"AFMoE layer {layer_index} requires standard nn.Linear self_attn.{name}"
                )
        for name in ("q_norm", "k_norm"):
            if getattr(attention, name, None) is None:
                raise UnsupportedAfmoeConfigurationError(
                    f"AFMoE layer {layer_index} does not expose self_attn.{name}"
                )

        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
        ):
            if getattr(layer, name, None) is None:
                raise UnsupportedAfmoeConfigurationError(
                    f"AFMoE layer {layer_index} does not expose {name}"
                )

    forbidden_parameters = [
        name
        for name, _ in model.named_parameters()
        if any(token in name for token in _FORBIDDEN_MOE_PARAMETER_TOKENS)
    ]
    if forbidden_parameters:
        preview = ", ".join(forbidden_parameters[:3])
        raise UnsupportedAfmoeConfigurationError(
            f"dense-only AFMoE source contains MoE parameters: {preview}"
        )

    parameters = list(model.parameters())
    if any(parameter.is_meta for parameter in parameters):
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE PR 1 requires materialized parameters, not meta tensors"
        )
    devices = _parameter_devices(parameters)
    if len(devices) > 1:
        formatted = ", ".join(sorted(str(device) for device in devices))
        raise UnsupportedAfmoeConfigurationError(
            "dense-only AFMoE PR 1 requires a single-device source model; "
            f"got {formatted}"
        )


__all__ = [
    "UnsupportedAfmoeConfigurationError",
    "assert_supported_dense_afmoe",
]
