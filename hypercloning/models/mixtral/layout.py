"""Mixtral layout detection helpers.

Transformers has used at least two Mixtral MoE layouts:

* legacy ModuleList experts with per-expert w1/w2/w3 Linear modules
* packed expert tensors with experts.gate_up_proj and experts.down_proj

The cloner keeps these checks isolated from the weight-copying logic.
"""

from __future__ import annotations

from typing import Any


PACKED_EXPERTS = "packed_experts"
LEGACY_MODULELIST_EXPERTS = "legacy_modulelist_experts"
UNKNOWN = "unknown"


def get_mixtral_moe(layer: Any) -> Any:
    """Return the Mixtral MoE block from a decoder layer."""

    if hasattr(layer, "block_sparse_moe"):
        return layer.block_sparse_moe
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise AssertionError("Could not locate Mixtral MoE block on decoder layer")


def detect_mixtral_moe_layout(moe: Any) -> str:
    """Identify the supported Transformers Mixtral expert layout."""

    experts = getattr(moe, "experts", None)
    if experts is None:
        return UNKNOWN
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return PACKED_EXPERTS
    try:
        first_expert = experts[0]
    except Exception:
        first_expert = None
    if first_expert is not None and all(hasattr(first_expert, name) for name in ("w1", "w2", "w3")):
        return LEGACY_MODULELIST_EXPERTS
    return UNKNOWN


__all__ = [
    "LEGACY_MODULELIST_EXPERTS",
    "PACKED_EXPERTS",
    "UNKNOWN",
    "detect_mixtral_moe_layout",
    "get_mixtral_moe",
]
