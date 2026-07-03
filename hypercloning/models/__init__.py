"""Model-specific HyperCloning integrations."""

from __future__ import annotations

from importlib import import_module

from hypercloning.models.specs import BUILTIN_CLONERS, ModelClonerSpec
from hypercloning.registry import REGISTERED_IMPORT_ERRORS, register_cloning_function


def register_model_cloner(spec: ModelClonerSpec) -> None:
    """Register one architecture cloner from a lazy import specification."""

    try:
        config_cls = getattr(import_module(spec.config_module), spec.config_name)
        clone_fn = getattr(import_module(spec.cloner_module), spec.cloner_name)
    except Exception as exc:  # pragma: no cover - optional dependency dependent
        REGISTERED_IMPORT_ERRORS[spec.key] = repr(exc)
    else:
        register_cloning_function(config_cls.__name__, clone_fn)


def register_builtin_cloners() -> None:
    """Register all built-in cloners whose optional dependencies are importable."""

    for spec in BUILTIN_CLONERS:
        register_model_cloner(spec)


__all__ = ["BUILTIN_CLONERS", "ModelClonerSpec", "register_builtin_cloners", "register_model_cloner"]
