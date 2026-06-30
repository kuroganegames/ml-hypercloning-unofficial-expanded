#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""HyperCloning public API."""

from __future__ import annotations

from importlib import import_module

from hypercloning.clone_api import CloneFunction, CloneRequest, CloneResult, call_legacy_clone_function, normalize_clone_output
from hypercloning.registry import (
    REGISTERED_CLONING_FUNCTIONS,
    REGISTERED_IMPORT_ERRORS,
    REGISTRATION_ERRORS,
    clone_model,
    get_cloning_function,
    register_cloning_function,
    unregister_cloning_function,
)
from hypercloning.trace import CloneOp, CloneTrace


def _register_optional_cloner(error_key: str, config_module: str, config_name: str, cloner_module: str, cloner_name: str) -> None:
    try:
        config_cls = getattr(import_module(config_module), config_name)
        clone_fn = getattr(import_module(cloner_module), cloner_name)
    except Exception as exc:  # pragma: no cover - optional dependency dependent
        REGISTERED_IMPORT_ERRORS[error_key] = repr(exc)
    else:
        register_cloning_function(config_cls.__name__, clone_fn)


def _try_register_builtin_cloners() -> None:
    for spec in (
        ("llama", "transformers", "LlamaConfig", "hypercloning.llama_cloning", "clone_llama"),
        ("gemma", "transformers", "GemmaConfig", "hypercloning.gemma_cloning", "clone_gemma"),
        ("gemma2", "transformers", "Gemma2Config", "hypercloning.gemma_cloning", "clone_gemma2"),
        ("opt", "transformers", "OPTConfig", "hypercloning.opt_cloning", "clone_opt"),
        ("pythia", "transformers", "GPTNeoXConfig", "hypercloning.pythia_cloning", "clone_pythia"),
        ("qwen3", "transformers", "Qwen3Config", "hypercloning.qwen3_cloning", "clone_qwen3"),
        ("olmo", "hf_olmo.configuration_olmo", "OLMoConfig", "hypercloning.olmo_cloning", "clone_olmo"),
    ):
        _register_optional_cloner(*spec)


_try_register_builtin_cloners()


def cloneModel(model, embedding_dim_multiplier: int, up_project_multiplier: int, **kwargs):
    cloning_function = get_cloning_function(model)
    print(f"cloning the network using {cloning_function} ...")
    return cloning_function(
        model,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
        **kwargs,
    )


def cloneModelWithResult(model, embedding_dim_multiplier: int, up_project_multiplier: int, **kwargs) -> CloneResult:
    output = cloneModel(model, embedding_dim_multiplier, up_project_multiplier, **kwargs)
    return normalize_clone_output(
        output,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
        num_heads_multiplier=kwargs.get("num_heads_multiplier", embedding_dim_multiplier),
        snr_db=kwargs.get("snr_db"),
    )


__all__ = [
    "CloneFunction",
    "CloneOp",
    "CloneRequest",
    "CloneResult",
    "CloneTrace",
    "REGISTERED_CLONING_FUNCTIONS",
    "REGISTERED_IMPORT_ERRORS",
    "REGISTRATION_ERRORS",
    "call_legacy_clone_function",
    "cloneModel",
    "clone_model",
    "cloneModelWithResult",
    "get_cloning_function",
    "normalize_clone_output",
    "register_cloning_function",
    "unregister_cloning_function",
]
