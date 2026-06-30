#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
"""Clone operation trace used by architecture-independent validators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

CloneOpKind = Literal[
    "matrix",
    "vector",
    "embedding",
    "linear",
    "layer_norm",
    "rms_norm",
    "qkv",
    "positional_embedding",
    "lm_head_scaled",
    "custom",
]


@dataclass
class CloneOp:
    kind: CloneOpKind
    src_param: str
    dst_param: str
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]
    normalize_input_repeat: bool = True
    repeat: Optional[tuple[int, ...]] = None
    scale: Optional[float] = None
    custom_validator: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CloneTrace:
    ops: list[CloneOp] = field(default_factory=list)
    config_mutations: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_op(self, op: CloneOp) -> CloneOp:
        self.ops.append(op)
        return op

    def add_matrix(
        self,
        src_param: str,
        dst_param: str,
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
        *,
        kind: CloneOpKind = "matrix",
        normalize_input_repeat: bool = True,
        scale: float | None = None,
        note: str = "",
    ) -> CloneOp:
        return self.add_op(
            CloneOp(
                kind=kind,
                src_param=src_param,
                dst_param=dst_param,
                src_shape=src_shape,
                dst_shape=dst_shape,
                normalize_input_repeat=normalize_input_repeat,
                scale=scale,
                note=note,
            )
        )

    def add_vector(
        self,
        src_param: str,
        dst_param: str,
        src_shape: tuple[int, ...],
        dst_shape: tuple[int, ...],
        *,
        kind: CloneOpKind = "vector",
        scale: float | None = None,
        note: str = "",
    ) -> CloneOp:
        return self.add_matrix(
            src_param,
            dst_param,
            src_shape,
            dst_shape,
            kind=kind,
            normalize_input_repeat=False,
            scale=scale,
            note=note,
        )

    def add_config_mutation(self, key: str, before: Any, after: Any) -> None:
        self.config_mutations[key] = (before, after)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ops": [op.to_dict() for op in self.ops],
            "config_mutations": dict(self.config_mutations),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
