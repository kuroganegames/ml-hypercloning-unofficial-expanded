"""Mixtral HyperCloning integration."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "clone_mixtral":
        from hypercloning.models.mixtral.cloning import clone_mixtral

        return clone_mixtral
    raise AttributeError(name)


__all__ = ["clone_mixtral"]
