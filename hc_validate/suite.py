"""P0 validation suite orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hc_validate import ValidationReport, ValidationSpec, audit_config, validate_forward
from hc_validate.inputs import BatchBundle, InputMatrixSpec, make_input_matrix, tokenizer_fingerprint
from hc_validate.serialization import SaveReloadSpec, run_save_reload_check


@dataclass
class P0ValidationSpec:
    base: ValidationSpec = field(default_factory=ValidationSpec)
    inputs: InputMatrixSpec = field(default_factory=InputMatrixSpec)
    save_reload: SaveReloadSpec = field(default_factory=SaveReloadSpec)
    run_config_audit: bool = True
    run_input_matrix_forward: bool = True
    run_save_reload: bool = True
    fail_on_empty_input_matrix: bool = True


def merge_child(parent: ValidationReport, prefix: str, child: ValidationReport) -> None:
    parent.metrics[prefix] = child.metrics
    parent.warnings.extend(f"{prefix}: {warning}" for warning in child.warnings)
    for failure in child.failures:
        parent.fail(f"{prefix}: {failure}")


def run_forward_on_bundles(source_model: Any, destination_model: Any, bundles: list[BatchBundle], spec: ValidationSpec) -> ValidationReport:
    report = ValidationReport(metrics={"num_bundles": len(bundles), "bundle_names": [bundle.name for bundle in bundles]})
    for bundle in bundles:
        if not bundle.batches:
            report.warnings.append(f"input bundle {bundle.name!r} has no batches: {bundle.metadata}")
            continue
        child = validate_forward(source_model, destination_model, bundle.batches, spec)
        merge_child(report, bundle.name, child)
    return report


def run_p0_validation(
    source_model: Any,
    destination_model: Any,
    *,
    tokenizer: Any | None = None,
    spec: P0ValidationSpec | None = None,
) -> ValidationReport:
    """Run P0 checks on an already cloned source/destination pair.

    P0 covers the most practical early checks: config multiplier audit, random
    and tokenizer input matrices, and save_pretrained/from_pretrained stability.
    """

    spec = spec or P0ValidationSpec()
    report = ValidationReport(metrics={"tokenizer_fingerprint": tokenizer_fingerprint(tokenizer)})
    bundles = make_input_matrix(tokenizer, getattr(source_model, "config", None), spec.inputs)
    report.metrics["input_bundles"] = [{"name": b.name, "num_batches": len(b.batches), "metadata": b.metadata} for b in bundles]

    nonempty_bundles = [bundle for bundle in bundles if bundle.batches]
    if not nonempty_bundles:
        msg = "input matrix produced no batches"
        if spec.fail_on_empty_input_matrix:
            report.fail(msg)
        else:
            report.warnings.append(msg)
        return report

    if spec.run_config_audit and hasattr(source_model, "config") and hasattr(destination_model, "config"):
        child = audit_config(source_model.config, destination_model.config, spec.base)
        merge_child(report, "config", child)

    if spec.run_input_matrix_forward:
        child = run_forward_on_bundles(source_model, destination_model, nonempty_bundles, spec.base)
        merge_child(report, "inputs", child)

    if spec.run_save_reload:
        # Use the first nonempty bundle to keep P0 serialization checks fast.
        child = run_save_reload_check(
            source_model,
            destination_model,
            nonempty_bundles[0].batches,
            spec.base,
            tokenizer=tokenizer,
            spec=spec.save_reload,
        )
        merge_child(report, "save_reload", child)

    return report
