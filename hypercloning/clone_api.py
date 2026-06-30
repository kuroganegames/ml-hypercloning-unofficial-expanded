#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Architecture-neutral API objects for HyperCloning integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar, Union, runtime_checkable

import torch

from hypercloning.trace import CloneTrace

ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class CloneRequest:
    """Requested width expansion for a HyperCloning plugin."""

    embedding_dim_multiplier: int
    up_project_multiplier: int
    num_heads_multiplier: Optional[int] = None
    snr_db: Optional[float] = None
    dtype: Optional[torch.dtype] = None
    device: Union[str, torch.device, None] = None
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_legacy_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.num_heads_multiplier is not None:
            kwargs["num_heads_multiplier"] = self.num_heads_multiplier
        if self.snr_db is not None:
            kwargs["snr_db"] = self.snr_db
        return kwargs


@dataclass
class CloneResult:
    """Normalized output from an architecture-specific cloner."""

    model: ModelT
    trace: CloneTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CloneFunction(Protocol[ModelT]):
    def __call__(self, source_model: ModelT, request: CloneRequest) -> CloneResult[ModelT] | ModelT:
        ...


def normalize_clone_output(output: CloneResult[ModelT] | ModelT, **metadata: Any) -> CloneResult[ModelT]:
    if isinstance(output, CloneResult):
        if metadata:
            output.metadata.update(metadata)
        return output
    return CloneResult(model=output, metadata=dict(metadata))


def call_legacy_clone_function(clone_fn, source_model, request: CloneRequest) -> CloneResult:
    output = clone_fn(
        source_model,
        embedding_dim_multiplier=request.embedding_dim_multiplier,
        up_project_multiplier=request.up_project_multiplier,
        **request.as_legacy_kwargs(),
    )
    return normalize_clone_output(
        output,
        embedding_dim_multiplier=request.embedding_dim_multiplier,
        up_project_multiplier=request.up_project_multiplier,
        num_heads_multiplier=request.num_heads_multiplier,
        snr_db=request.snr_db,
    )
