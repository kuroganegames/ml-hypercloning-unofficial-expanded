"""Checkpoint-tier P0 validation.

This module lets CI run tiny/local cases cheaply while allowing real checkpoint
cases to be enabled explicitly with an environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import os
from typing import Any

import torch

from hc_validate import ValidationReport, ValidationSpec, audit_config, validate_forward
from hc_validate.inputs import BatchBundle, InputMatrixSpec, make_input_matrix, tokenizer_fingerprint
from hc_validate.serialization import SaveReloadSpec, run_save_reload_check
from hypercloning.clone_api import CloneRequest, CloneResult, normalize_clone_output


@dataclass
class CheckpointCase:
    name: str
    source: str | None = None
    tokenizer: str | None = None
    clone_entry: str | None = None
    config_factory: str | None = None
    model_factory: str | None = None
    trust_remote_code: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    enabled: bool = True
    requires_env: str | None = None
    input_spec: InputMatrixSpec | None = None
    save_reload_spec: SaveReloadSpec | None = None


@dataclass
class CheckpointMatrixSpec:
    cases: tuple[CheckpointCase, ...] = ()
    run_forward: bool = True
    run_config_audit: bool = True
    run_save_reload: bool = True
    fail_on_skipped_required_case: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def import_object(entry: str) -> Any:
    if ":" in entry:
        module_name, attr = entry.split(":", 1)
    else:
        module_name, attr = entry.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def torch_dtype_from_name(name: str) -> torch.dtype:
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


def _load_tokenizer(case: CheckpointCase) -> Any | None:
    if not case.tokenizer and not case.source:
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    target = case.tokenizer or case.source
    try:
        return AutoTokenizer.from_pretrained(target, trust_remote_code=case.trust_remote_code)
    except Exception:
        return None


def _load_source_model(case: CheckpointCase) -> Any:
    dtype = torch_dtype_from_name(case.dtype)
    if case.model_factory:
        factory = import_object(case.model_factory)
        return factory()
    if case.config_factory:
        factory = import_object(case.config_factory)
        config_or_model = factory()
        if hasattr(config_or_model, "forward") and hasattr(config_or_model, "state_dict"):
            return config_or_model
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_config(config_or_model, trust_remote_code=case.trust_remote_code)
    if not case.source:
        raise ValueError(f"CheckpointCase {case.name!r} must define source, config_factory, or model_factory")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        case.source,
        trust_remote_code=case.trust_remote_code,
        torch_dtype=dtype,
    )


def _clone_source(source_model: Any, case: CheckpointCase, spec: ValidationSpec) -> CloneResult:
    if not case.clone_entry:
        raise ValueError(f"CheckpointCase {case.name!r} requires clone_entry")
    clone_fn = import_object(case.clone_entry)
    request = CloneRequest(
        embedding_dim_multiplier=spec.embedding_dim_multiplier,
        up_project_multiplier=spec.up_project_multiplier,
        num_heads_multiplier=spec.num_heads_multiplier,
        dtype=torch_dtype_from_name(case.dtype),
        device=case.device,
        trust_remote_code=case.trust_remote_code,
    )
    try:
        output = clone_fn(source_model, request)
    except TypeError:
        kwargs = request.as_legacy_kwargs()
        output = clone_fn(
            source_model,
            embedding_dim_multiplier=request.embedding_dim_multiplier,
            up_project_multiplier=request.up_project_multiplier,
            **kwargs,
        )
    return normalize_clone_output(output, checkpoint_case=case.name)


def _merge_child(parent: ValidationReport, prefix: str, child: ValidationReport) -> None:
    parent.metrics[prefix] = child.metrics
    parent.warnings.extend(f"{prefix}: {w}" for w in child.warnings)
    for failure in child.failures:
        parent.fail(f"{prefix}: {failure}")


def _run_bundled_forward(source_model: Any, destination_model: Any, bundles: list[BatchBundle], spec: ValidationSpec) -> ValidationReport:
    report = ValidationReport(metrics={"bundles": [b.name for b in bundles], "num_bundles": len(bundles)})
    for bundle in bundles:
        if not bundle.batches:
            report.warnings.append(f"input bundle {bundle.name!r} produced no batches: {bundle.metadata}")
            continue
        child = validate_forward(source_model, destination_model, bundle.batches, spec)
        _merge_child(report, f"forward.{bundle.name}", child)
    return report


def run_checkpoint_case(case: CheckpointCase, base_spec: ValidationSpec, matrix_spec: CheckpointMatrixSpec | None = None) -> ValidationReport:
    matrix_spec = matrix_spec or CheckpointMatrixSpec(cases=(case,))
    report = ValidationReport(metrics={"case": case.name, "enabled": case.enabled})
    if not case.enabled:
        report.warnings.append("checkpoint case disabled")
        return report
    if case.requires_env and not os.environ.get(case.requires_env):
        msg = f"checkpoint case skipped because {case.requires_env} is not set"
        report.metrics["skipped"] = True
        if matrix_spec.fail_on_skipped_required_case:
            report.fail(msg)
        else:
            report.warnings.append(msg)
        return report

    try:
        spec = replace(base_spec, device=case.device)
        source_model = _load_source_model(case)
        if hasattr(source_model, "to"):
            source_model.to(case.device)
        tokenizer = _load_tokenizer(case)
        clone_result = _clone_source(source_model, case, spec)
        destination_model = clone_result.model
        if hasattr(destination_model, "to"):
            destination_model.to(case.device)

        report.metrics["tokenizer_fingerprint"] = tokenizer_fingerprint(tokenizer)
        input_spec = case.input_spec or InputMatrixSpec(
            seq_lengths=(min(base_spec.seq_len, 128),),
            batch_size=base_spec.batch_size,
            num_random_batches_per_length=base_spec.num_batches,
            seed=base_spec.seed,
        )
        bundles = make_input_matrix(tokenizer, getattr(source_model, "config", None), input_spec)

        if matrix_spec.run_config_audit:
            child = audit_config(source_model.config, destination_model.config, spec)
            _merge_child(report, "config", child)
        if matrix_spec.run_forward:
            child = _run_bundled_forward(source_model, destination_model, bundles, spec)
            _merge_child(report, "inputs", child)
        if matrix_spec.run_save_reload:
            first_batches = next((bundle.batches for bundle in bundles if bundle.batches), [])
            save_spec = case.save_reload_spec or SaveReloadSpec(trust_remote_code=case.trust_remote_code)
            child = run_save_reload_check(
                source_model,
                destination_model,
                first_batches,
                spec,
                tokenizer=tokenizer,
                spec=save_spec,
            )
            _merge_child(report, "save_reload", child)
    except Exception as exc:
        report.fail(f"checkpoint case {case.name!r} raised {type(exc).__name__}: {exc}")
    return report


def run_checkpoint_matrix(matrix_spec: CheckpointMatrixSpec, base_spec: ValidationSpec | None = None) -> ValidationReport:
    base_spec = base_spec or ValidationSpec()
    report = ValidationReport(metrics={"num_cases": len(matrix_spec.cases), "metadata": matrix_spec.metadata})
    for case in matrix_spec.cases:
        child = run_checkpoint_case(case, base_spec, matrix_spec)
        _merge_child(report, f"checkpoint.{case.name}", child)
    return report
