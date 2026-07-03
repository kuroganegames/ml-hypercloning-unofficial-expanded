"""Hugging Face model loading helpers for validation runners."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ModelLoadSpec:
    """Options for loading one Hugging Face causal LM."""

    trust_remote_code: bool = False
    revision: str | None = None
    dtype: str | None = "auto"

    # Single-device mode. Ignored when device_map is provided.
    device: str | None = None

    # Large-model / Accelerate mode.
    device_map: str | dict[str, Any] | None = None
    max_memory: dict[str, str] | None = None
    low_cpu_mem_usage: bool = False
    offload_folder: str | None = None
    offload_state_dict: bool = False

    # Kernel / attention behavior.
    attn_implementation: str | None = None

    # Offline / reproducibility.
    local_files_only: bool = False

    # Escape hatch for architecture-specific or transformers-version-specific kwargs.
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


def torch_dtype_from_name(name: str | None):
    if name is None or name.lower() in {"auto", "none"}:
        return "auto"

    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = name.lower()
    if key not in aliases:
        raise ValueError(f"Unsupported dtype {name!r}")
    return aliases[key]


def load_json_arg(value: str | None):
    """Load a JSON CLI argument, accepting inline JSON or @path syntax."""

    if not value:
        return None
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def build_from_pretrained_kwargs(spec: ModelLoadSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": spec.local_files_only,
    }

    if spec.revision:
        kwargs["revision"] = spec.revision

    torch_dtype = torch_dtype_from_name(spec.dtype)
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype

    if spec.device_map is not None:
        kwargs["device_map"] = spec.device_map
    if spec.max_memory is not None:
        kwargs["max_memory"] = spec.max_memory
    if spec.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = True
    if spec.offload_folder:
        kwargs["offload_folder"] = spec.offload_folder
    if spec.offload_state_dict:
        kwargs["offload_state_dict"] = True
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation

    kwargs.update(spec.extra_kwargs)
    return kwargs


def load_hf_causal_lm(model_id_or_path: str, spec: ModelLoadSpec):
    """Load a causal LM from a Hub ID or local directory.

    When ``device_map`` is provided, Accelerate owns module placement. In that
    mode this helper intentionally does not call ``model.to(...)``.
    """

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **build_from_pretrained_kwargs(spec))
    if spec.device_map is not None:
        return model
    return model.to(spec.device) if spec.device and hasattr(model, "to") else model


def model_uses_device_map(model: Any) -> bool:
    return bool(getattr(model, "hf_device_map", None))


def maybe_move_model(model: Any, device: str | None):
    """Move a model only when it is not already Accelerate-dispatched."""

    model.eval()
    if device and not model_uses_device_map(model) and hasattr(model, "to"):
        return model.to(device)
    return model


def resolve_model_input_device(model: Any, fallback: str | torch.device | None = "cpu") -> torch.device:
    """Find a safe device for input tensors for a possibly sharded model."""

    try:
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and not getattr(weight, "is_meta", False):
            return weight.device
    except Exception:
        pass

    for param in model.parameters():
        if not getattr(param, "is_meta", False):
            return param.device

    return torch.device(fallback or "cpu")


def move_batch_to_model(batch: dict[str, Any], model: Any, fallback: str | torch.device | None = "cpu") -> dict[str, Any]:
    device = resolve_model_input_device(model, fallback=fallback)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


__all__ = [
    "ModelLoadSpec",
    "build_from_pretrained_kwargs",
    "load_hf_causal_lm",
    "load_json_arg",
    "maybe_move_model",
    "model_uses_device_map",
    "move_batch_to_model",
    "resolve_model_input_device",
    "torch_dtype_from_name",
]
