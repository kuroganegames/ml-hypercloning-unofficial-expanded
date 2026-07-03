#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""HyperCloning public API."""

from __future__ import annotations

from hypercloning.clone_api import CloneFunction, CloneRequest, CloneResult, call_legacy_clone_function, normalize_clone_output
from hypercloning.models import register_builtin_cloners
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


register_builtin_cloners()


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
