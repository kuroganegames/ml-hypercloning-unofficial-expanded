#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Runtime registry for HyperCloning implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CloneCallable = Callable[..., Any]

REGISTERED_CLONING_FUNCTIONS: dict[str, CloneCallable] = {}
REGISTERED_IMPORT_ERRORS: dict[str, str] = {}
REGISTRATION_ERRORS = REGISTERED_IMPORT_ERRORS


def register_cloning_function(config_class_name: str, clone_fn: CloneCallable, *, overwrite: bool = True) -> None:
    if not config_class_name:
        raise ValueError("config_class_name must be non-empty")
    if not overwrite and config_class_name in REGISTERED_CLONING_FUNCTIONS:
        raise KeyError(f"cloner already registered for {config_class_name!r}")
    REGISTERED_CLONING_FUNCTIONS[config_class_name] = clone_fn


def unregister_cloning_function(config_class_name: str) -> None:
    REGISTERED_CLONING_FUNCTIONS.pop(config_class_name, None)


def get_config_class_name(model_or_config: Any) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    return type(config).__name__


def get_cloning_function(model_or_config: Any) -> CloneCallable:
    key = get_config_class_name(model_or_config)
    if key not in REGISTERED_CLONING_FUNCTIONS:
        available = ", ".join(sorted(REGISTERED_CLONING_FUNCTIONS)) or "<none>"
        import_errors = "; ".join(
            f"{name}: {message}" for name, message in sorted(REGISTERED_IMPORT_ERRORS.items())
        )
        suffix = f" Import errors: {import_errors}" if import_errors else ""
        raise AssertionError(
            f"cloning is not supported for model config of type {key}. "
            f"Available config classes: {available}.{suffix}"
        )
    return REGISTERED_CLONING_FUNCTIONS[key]


def clone_model(model: Any, embedding_dim_multiplier: int, up_project_multiplier: int, **kwargs: Any) -> Any:
    return get_cloning_function(model)(
        model,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
        **kwargs,
    )


register_cloner = register_cloning_function
get_cloner_for_model = get_cloning_function
resolve_cloner = lambda config_class_name: REGISTERED_CLONING_FUNCTIONS[config_class_name]
