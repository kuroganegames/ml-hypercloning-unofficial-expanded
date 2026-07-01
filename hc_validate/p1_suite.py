"""P1 validation suite orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hc_validate import ValidationReport, ValidationSpec
from hc_validate.generation import GenerationCacheSpec, run_generation_cache_check
from hc_validate.inputs import InputMatrixSpec, make_input_matrix
from hc_validate.long_context import LongContextSpec, run_long_context_check


@dataclass
class P1ValidationSpec:
    base: ValidationSpec = field(default_factory=lambda: ValidationSpec(compare_hidden_states=False))
    prompt_inputs: InputMatrixSpec = field(
        default_factory=lambda: InputMatrixSpec(
            seq_lengths=(8, 32, 128),
            num_random_batches_per_length=1,
            tokenizer_fixture_batches=True,
            chat_template_batches=True,
        )
    )
    generation: GenerationCacheSpec = field(default_factory=GenerationCacheSpec)
    long_context: LongContextSpec = field(default_factory=LongContextSpec)
    run_generation: bool = True
    run_long_context: bool = True
    fail_on_empty_generation_inputs: bool = True
    fail_fast: bool = False


def merge_child(parent: ValidationReport, prefix: str, child: ValidationReport) -> None:
    parent.metrics[prefix] = child.metrics
    parent.warnings.extend(f"{prefix}: {warning}" for warning in child.warnings)
    for failure in child.failures:
        parent.fail(f"{prefix}: {failure}")


def _first_nonempty_batches(bundles) -> list[dict[str, Any]]:
    batches = []
    for bundle in bundles:
        batches.extend(bundle.batches)
    return batches


def run_p1_validation(
    source_model: Any,
    destination_model: Any,
    *,
    tokenizer: Any | None = None,
    spec: P1ValidationSpec | None = None,
) -> ValidationReport:
    """Run P1 checks on an already cloned source/destination pair.

    P1 covers generation/cache behavior and long-context position/mask behavior.
    It keeps the validator architecture-independent by relying on public
    ``forward``/``generate``/``config`` surfaces only.
    """

    spec = spec or P1ValidationSpec()
    report = ValidationReport(metrics={"p1": True})

    if spec.run_generation:
        try:
            bundles = make_input_matrix(tokenizer, getattr(source_model, "config", None), spec.prompt_inputs)
            batches = _first_nonempty_batches(bundles)
            report.metrics["generation_input_bundles"] = [
                {"name": bundle.name, "num_batches": len(bundle.batches), "metadata": bundle.metadata}
                for bundle in bundles
            ]
            if not batches:
                message = "P1 generation input matrix produced no batches"
                if spec.fail_on_empty_generation_inputs:
                    report.fail(message)
                else:
                    report.warnings.append(message)
            else:
                child = run_generation_cache_check(
                    source_model,
                    destination_model,
                    batches,
                    spec.base,
                    spec.generation,
                )
                merge_child(report, "generation", child)
        except Exception as exc:
            report.fail(f"generation P1 check raised {type(exc).__name__}: {exc}")
        if spec.fail_fast and not report.ok:
            return report

    if spec.run_long_context:
        try:
            child = run_long_context_check(
                source_model,
                destination_model,
                tokenizer=tokenizer,
                base_spec=spec.base,
                long_spec=spec.long_context,
                generation_spec=spec.generation if spec.long_context.run_generation_cache_at_long_context else None,
            )
            merge_child(report, "long_context", child)
        except Exception as exc:
            report.fail(f"long-context P1 check raised {type(exc).__name__}: {exc}")

    return report
