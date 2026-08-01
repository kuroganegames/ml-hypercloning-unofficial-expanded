"""Dense-only AFMoE HyperCloning integration."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {
        "clone_afmoe",
        "clone_afmoe_attention",
        "clone_afmoe_rms_norm",
    }:
        from hypercloning.models.afmoe.cloning import (
            clone_afmoe,
            clone_afmoe_attention,
            clone_afmoe_rms_norm,
        )

        return {
            "clone_afmoe": clone_afmoe,
            "clone_afmoe_attention": clone_afmoe_attention,
            "clone_afmoe_rms_norm": clone_afmoe_rms_norm,
        }[name]
    if name in {
        "UnsupportedAfmoeConfigurationError",
        "assert_supported_dense_afmoe",
    }:
        from hypercloning.models.afmoe.guards import (
            UnsupportedAfmoeConfigurationError,
            assert_supported_dense_afmoe,
        )

        return {
            "UnsupportedAfmoeConfigurationError": UnsupportedAfmoeConfigurationError,
            "assert_supported_dense_afmoe": assert_supported_dense_afmoe,
        }[name]
    raise AttributeError(name)


__all__ = [
    "UnsupportedAfmoeConfigurationError",
    "assert_supported_dense_afmoe",
    "clone_afmoe",
    "clone_afmoe_attention",
    "clone_afmoe_rms_norm",
]
