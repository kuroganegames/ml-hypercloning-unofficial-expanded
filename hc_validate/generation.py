"""P1 generation and cache-path validation.

The checks in this module treat a Hugging Face-style causal LM as a black box.
They validate generation/cache behavior through public ``forward`` and optional
``generate`` APIs instead of architecture-specific module paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any

import torch

from hc_validate import ValidationReport, ValidationSpec, tensor_diff


@dataclass
class GenerationCacheSpec:
    enabled: bool = True
    max_new_tokens: int = 8
    num_prompt_batches: int | None = None

    run_generate: bool = True
    compare_sequences: bool = True
    compare_generate_logits: bool = True

    run_manual_cache_loop: bool = True
    compare_prefill_logits: bool = True
    compare_step_logits: bool = True
    teacher_forced_next_tokens: bool = True

    run_no_cache_vs_cache: bool = True

    do_sample: bool = False
    num_beams: int = 1
    use_cache: bool = True
    cache_implementation: str | None = None
    output_logits: bool = True
    output_scores: bool = True

    atol: float = 1e-4
    rtol: float = 1e-4
    require_generate_support: bool = False
    require_past_key_values: bool = False
    max_failures_per_batch: int = 3


@dataclass
class ManualCacheLoopResult:
    logits: list[torch.Tensor]
    tokens: torch.Tensor
    past_key_values_summary: dict[str, Any] | None = None


def _merge_child(parent: ValidationReport, prefix: str, child: ValidationReport) -> None:
    parent.metrics[prefix] = child.metrics
    parent.warnings.extend(f"{prefix}: {warning}" for warning in child.warnings)
    for failure in child.failures:
        parent.fail(f"{prefix}: {failure}")


def _accepted_kwargs(callable_obj: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_obj).parameters)
    except Exception:
        return set()


def _filter_forward_kwargs(model: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = _accepted_kwargs(model.forward)
    if not accepted:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _get_output_attr(output: Any, name: str, default: Any = None) -> Any:
    if hasattr(output, name):
        return getattr(output, name)
    if isinstance(output, dict):
        return output.get(name, default)
    return default


def _last_nonpad_logits(logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return logits[:, -1, :]
    lengths = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, lengths, :]


def _top1_match_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean().item())


def _compare_logits(name: str, src: torch.Tensor, dst: torch.Tensor, spec: GenerationCacheSpec) -> ValidationReport:
    report = ValidationReport()
    diff = tensor_diff(src, dst)
    report.metrics.update(
        {
            "max_abs": diff.max_abs,
            "mean_abs": diff.mean_abs,
            "rel_l2": diff.rel_l2,
            "cosine": diff.cosine,
            "top1_match_rate": _top1_match_rate(src, dst),
        }
    )
    if diff.max_abs > spec.atol and diff.rel_l2 > spec.rtol:
        report.fail(f"{name}: logits mismatch max_abs={diff.max_abs:.3g} rel_l2={diff.rel_l2:.3g}")
    return report


def _summarize_past(past: Any) -> dict[str, Any] | None:
    if past is None:
        return None
    if torch.is_tensor(past):
        return {"type": "Tensor", "shape": tuple(past.shape), "dtype": str(past.dtype)}
    if isinstance(past, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in past.keys())}
    if isinstance(past, (list, tuple)):
        summary: dict[str, Any] = {"type": type(past).__name__, "length": len(past)}
        if past:
            first = past[0]
            if isinstance(first, (list, tuple)):
                summary["first_entry_length"] = len(first)
                if first and torch.is_tensor(first[0]):
                    summary["first_tensor_shape"] = tuple(first[0].shape)
            elif torch.is_tensor(first):
                summary["first_tensor_shape"] = tuple(first.shape)
        return summary
    return {"type": type(past).__name__}


@torch.inference_mode()
def manual_cache_loop(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    steps: int,
    device: str,
    forced_tokens: torch.Tensor | None = None,
) -> ManualCacheLoopResult:
    model.eval().to(device)
    batch = _move_batch(batch, device)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    prefill_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    prefill_out = model(**_filter_forward_kwargs(model, prefill_kwargs))
    logits = _get_output_attr(prefill_out, "logits")
    if logits is None:
        raise RuntimeError("manual cache loop requires model.forward to return logits")
    past = _get_output_attr(prefill_out, "past_key_values")
    logits_per_step = [_last_nonpad_logits(logits, attention_mask)]

    generated_tokens: list[torch.Tensor] = []
    next_token = logits_per_step[-1].argmax(dim=-1, keepdim=True)
    if forced_tokens is not None and forced_tokens.numel() > 0:
        next_token = forced_tokens[:, 0:1].to(device)

    extended_mask = attention_mask
    for step in range(steps):
        decode_mask = torch.ones_like(next_token)
        extended_mask = torch.cat([extended_mask, decode_mask], dim=-1)
        decode_kwargs = {
            "input_ids": next_token,
            "attention_mask": extended_mask,
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        out = model(**_filter_forward_kwargs(model, decode_kwargs))
        step_logits = _get_output_attr(out, "logits")
        if step_logits is None:
            raise RuntimeError("manual cache decode requires model.forward to return logits")
        past = _get_output_attr(out, "past_key_values")
        logits_per_step.append(step_logits[:, -1, :])
        generated_tokens.append(next_token)

        if forced_tokens is not None and step + 1 < forced_tokens.shape[1]:
            next_token = forced_tokens[:, step + 1 : step + 2].to(device)
        else:
            next_token = step_logits[:, -1, :].argmax(dim=-1, keepdim=True)

    tokens = torch.cat(generated_tokens, dim=-1) if generated_tokens else input_ids.new_zeros((input_ids.shape[0], 0))
    return ManualCacheLoopResult(logits=logits_per_step, tokens=tokens, past_key_values_summary=_summarize_past(past))


def _full_forward_last_logits(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, device: str) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device) if attention_mask is not None else None,
        "use_cache": False,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    out = model(**_filter_forward_kwargs(model, kwargs))
    logits = _get_output_attr(out, "logits")
    if logits is None:
        raise RuntimeError("full forward requires logits")
    return _last_nonpad_logits(logits, kwargs.get("attention_mask"))


def _run_no_cache_vs_cache(
    model: Any,
    batch: dict[str, torch.Tensor],
    loop: ManualCacheLoopResult,
    spec: GenerationCacheSpec,
    base_spec: ValidationSpec,
    label: str,
) -> ValidationReport:
    report = ValidationReport(metrics={"label": label})
    batch = _move_batch(batch, base_spec.device)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    for step in range(loop.tokens.shape[1]):
        prefix = torch.cat([input_ids, loop.tokens[:, : step + 1].to(input_ids.device)], dim=-1)
        extra_mask = torch.ones_like(loop.tokens[:, : step + 1]).to(attention_mask.device)
        prefix_mask = torch.cat([attention_mask, extra_mask], dim=-1)
        full_logits = _full_forward_last_logits(model, prefix, prefix_mask, base_spec.device)
        cache_logits = loop.logits[step + 1]
        child = _compare_logits(f"{label}.no_cache_vs_cache.step{step}", full_logits, cache_logits, spec)
        _merge_child(report, f"step{step}", child)
    return report


def _generate_kwargs(spec: GenerationCacheSpec) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": spec.max_new_tokens,
        "do_sample": spec.do_sample,
        "num_beams": spec.num_beams,
        "use_cache": spec.use_cache,
        "return_dict_in_generate": True,
        "output_scores": spec.output_scores,
    }
    if spec.output_logits:
        kwargs["output_logits"] = True
    if spec.cache_implementation is not None:
        kwargs["cache_implementation"] = spec.cache_implementation
    return kwargs


def _call_generate(model: Any, batch: dict[str, torch.Tensor], spec: GenerationCacheSpec) -> tuple[Any | None, str | None]:
    if not hasattr(model, "generate"):
        return None, "model does not implement generate"
    kwargs = _generate_kwargs(spec)
    allowed = _accepted_kwargs(model.generate)
    if allowed:
        call_kwargs = {key: value for key, value in kwargs.items() if key in allowed or key in {"max_new_tokens", "do_sample", "num_beams"}}
    else:
        call_kwargs = kwargs
    model_inputs = {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
    try:
        return model.generate(**model_inputs, **call_kwargs), None
    except TypeError as exc:
        if "output_logits" in call_kwargs:
            call_kwargs.pop("output_logits", None)
            try:
                return model.generate(**model_inputs, **call_kwargs), "generate retried without output_logits"
            except Exception as retry_exc:
                return None, f"generate failed after retry: {retry_exc!r}; original={exc!r}"
        return None, f"generate failed: {exc!r}"
    except Exception as exc:
        return None, f"generate failed: {exc!r}"


def _extract_generate_step_logits(output: Any) -> tuple[list[torch.Tensor], str]:
    logits = _get_output_attr(output, "logits")
    if logits is not None:
        return list(logits), "logits"
    scores = _get_output_attr(output, "scores")
    if scores is not None:
        return list(scores), "scores"
    return [], "none"


def run_generate_compare(
    source_model: Any,
    destination_model: Any,
    batch: dict[str, torch.Tensor],
    base_spec: ValidationSpec,
    spec: GenerationCacheSpec | None = None,
) -> ValidationReport:
    spec = spec or GenerationCacheSpec()
    report = ValidationReport(metrics={"enabled": spec.enabled, "max_new_tokens": spec.max_new_tokens})
    if not spec.enabled or not spec.run_generate:
        report.warnings.append("generate check disabled")
        return report
    batch = _move_batch(batch, base_spec.device)
    source_model.eval().to(base_spec.device)
    destination_model.eval().to(base_spec.device)

    src_out, src_error = _call_generate(source_model, batch, spec)
    dst_out, dst_error = _call_generate(destination_model, batch, spec)
    if src_error:
        (report.fail if spec.require_generate_support else report.warnings.append)(f"source generate: {src_error}")
    if dst_error:
        (report.fail if spec.require_generate_support else report.warnings.append)(f"destination generate: {dst_error}")
    if src_out is None or dst_out is None:
        return report

    src_seq = _get_output_attr(src_out, "sequences", src_out if torch.is_tensor(src_out) else None)
    dst_seq = _get_output_attr(dst_out, "sequences", dst_out if torch.is_tensor(dst_out) else None)
    if spec.compare_sequences and torch.is_tensor(src_seq) and torch.is_tensor(dst_seq):
        same = bool(torch.equal(src_seq, dst_seq))
        report.metrics["sequence_equal"] = same
        report.metrics["source_sequence_shape"] = tuple(src_seq.shape)
        report.metrics["destination_sequence_shape"] = tuple(dst_seq.shape)
        if not same:
            report.fail("generate sequence mismatch")

    if spec.compare_generate_logits:
        src_logits, src_kind = _extract_generate_step_logits(src_out)
        dst_logits, dst_kind = _extract_generate_step_logits(dst_out)
        report.metrics["generate_logits_kind"] = {"source": src_kind, "destination": dst_kind}
        if src_kind != dst_kind:
            report.warnings.append(f"generate output kind mismatch: source={src_kind} destination={dst_kind}")
        if src_logits and dst_logits:
            for step, (src_step, dst_step) in enumerate(zip(src_logits, dst_logits)):
                child = _compare_logits(f"generate.step{step}", src_step, dst_step, spec)
                _merge_child(report, f"step{step}", child)
    return report


def run_manual_cache_compare(
    source_model: Any,
    destination_model: Any,
    batch: dict[str, torch.Tensor],
    base_spec: ValidationSpec,
    spec: GenerationCacheSpec | None = None,
) -> ValidationReport:
    spec = spec or GenerationCacheSpec()
    report = ValidationReport(metrics={"enabled": spec.enabled, "max_new_tokens": spec.max_new_tokens})
    if not spec.enabled or not spec.run_manual_cache_loop:
        report.warnings.append("manual cache loop check disabled")
        return report
    try:
        src_loop = manual_cache_loop(source_model, batch, steps=spec.max_new_tokens, device=base_spec.device)
        forced = src_loop.tokens if spec.teacher_forced_next_tokens else None
        dst_loop = manual_cache_loop(destination_model, batch, steps=spec.max_new_tokens, device=base_spec.device, forced_tokens=forced)
        report.metrics["source_past"] = src_loop.past_key_values_summary
        report.metrics["destination_past"] = dst_loop.past_key_values_summary
        if spec.require_past_key_values and (src_loop.past_key_values_summary is None or dst_loop.past_key_values_summary is None):
            report.fail("past_key_values were required but missing from source or destination")
    except Exception as exc:
        report.fail(f"manual cache loop raised {type(exc).__name__}: {exc}")
        return report

    for step, (src_logits, dst_logits) in enumerate(zip(src_loop.logits, dst_loop.logits)):
        if step == 0 and not spec.compare_prefill_logits:
            continue
        if step > 0 and not spec.compare_step_logits:
            continue
        child = _compare_logits(f"manual_cache.step{step}", src_logits, dst_logits, spec)
        _merge_child(report, f"step{step}", child)

    if spec.run_no_cache_vs_cache:
        src_child = _run_no_cache_vs_cache(source_model, batch, src_loop, spec, base_spec, "source")
        dst_child = _run_no_cache_vs_cache(destination_model, batch, dst_loop, spec, base_spec, "destination")
        _merge_child(report, "source_no_cache_vs_cache", src_child)
        _merge_child(report, "destination_no_cache_vs_cache", dst_child)
    return report


def run_generation_cache_check(
    source_model: Any,
    destination_model: Any,
    batches: list[dict[str, torch.Tensor]],
    base_spec: ValidationSpec,
    spec: GenerationCacheSpec | None = None,
) -> ValidationReport:
    spec = spec or GenerationCacheSpec()
    report = ValidationReport(metrics={"enabled": spec.enabled, "num_batches": len(batches)})
    if not spec.enabled:
        report.warnings.append("generation/cache check disabled")
        return report
    selected = batches if spec.num_prompt_batches is None else batches[: spec.num_prompt_batches]
    for batch_index, batch in enumerate(selected):
        if spec.run_generate:
            child = run_generate_compare(source_model, destination_model, batch, base_spec, spec)
            _merge_child(report, f"batch{batch_index}.generate", child)
        if spec.run_manual_cache_loop:
            child = run_manual_cache_compare(source_model, destination_model, batch, base_spec, spec)
            _merge_child(report, f"batch{batch_index}.manual_cache", child)
    return report
