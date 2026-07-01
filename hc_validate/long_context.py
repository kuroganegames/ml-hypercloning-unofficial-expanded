"""P1 long-context, position-id, and attention-mask validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Literal

import torch

from hc_validate import ValidationReport, ValidationSpec, tensor_diff
from hc_validate.generation import GenerationCacheSpec, run_generation_cache_check
from hc_validate.inputs import infer_pad_token_id, infer_vocab_size

MaskPattern = Literal[
    "no_padding",
    "right_padding",
    "left_padding",
    "ragged",
    "same_token",
    "alternating",
    "explicit_position_ids",
    "sliding_window_boundary",
]


@dataclass
class LongContextSpec:
    enabled: bool = True
    lengths: tuple[int, ...] | None = None
    include_inferred_lengths: bool = True
    max_length_cap: int | None = 4096
    min_length: int = 1
    batch_size: int = 2
    seed: int = 0
    patterns: tuple[MaskPattern, ...] = (
        "no_padding",
        "right_padding",
        "left_padding",
        "ragged",
        "same_token",
        "alternating",
        "explicit_position_ids",
    )
    compare_all_logits_below_length: int = 256
    compare_last_token_logits: bool = True
    compare_nonpad_logits: bool = True
    selected_positions: tuple[int, ...] = (0, -1)
    run_generation_cache_at_long_context: bool = True
    generation_max_new_tokens: int = 4
    max_generation_bundles: int = 3
    atol: float = 1e-4
    rtol: float = 1e-4
    require_position_ids_support: bool = False


@dataclass
class LongContextBundle:
    name: str
    length: int
    pattern: MaskPattern
    batch: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _get_config_int(config: Any | None, names: tuple[str, ...]) -> int | None:
    if config is None:
        return None
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if isinstance(config, dict):
        for name in names:
            value = config.get(name)
            if isinstance(value, int) and value > 0:
                return value
    return None


def infer_context_lengths(config: Any | None, spec: LongContextSpec | None = None) -> tuple[int, ...]:
    spec = spec or LongContextSpec()
    candidates: set[int] = set()
    if spec.include_inferred_lengths:
        candidates.update({1, 2, 3, 8, 16, 64, 128})
        max_pos = _get_config_int(config, ("max_position_embeddings", "max_sequence_length", "seq_length", "n_positions"))
        if max_pos:
            candidates.update({max(1, max_pos // 4), max(1, max_pos // 2), max(1, max_pos - 1), max_pos})
        sliding = _get_config_int(config, ("sliding_window",))
        if sliding:
            candidates.update({max(1, sliding - 1), sliding, sliding + 1, sliding * 2})
    if spec.lengths:
        candidates.update(spec.lengths)
    if spec.max_length_cap is not None:
        candidates = {length for length in candidates if length <= spec.max_length_cap}
    return tuple(sorted(length for length in candidates if length >= spec.min_length))


def make_position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def _labels_from_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return input_ids.clone().masked_fill(attention_mask == 0, -100)


def _base_ids(vocab_size: int, batch_size: int, length: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(1, max(vocab_size, 2), (batch_size, length), generator=generator, dtype=torch.long)


def _make_pattern_batch(
    pattern: MaskPattern,
    *,
    length: int,
    vocab_size: int,
    pad_token_id: int,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    input_ids = _base_ids(vocab_size, batch_size, length, generator)
    attention_mask = torch.ones_like(input_ids)

    if pattern == "right_padding":
        pad_len = max(1, length // 4)
        input_ids[:, -pad_len:] = pad_token_id
        attention_mask[:, -pad_len:] = 0
    elif pattern == "left_padding":
        pad_len = max(1, length // 4)
        input_ids[:, :pad_len] = pad_token_id
        attention_mask[:, :pad_len] = 0
    elif pattern == "ragged":
        for row in range(batch_size):
            pad_len = min(length - 1, row + 1) if length > 1 else 0
            if pad_len:
                input_ids[row, -pad_len:] = pad_token_id
                attention_mask[row, -pad_len:] = 0
    elif pattern == "same_token":
        token_id = 1 if vocab_size <= 2 else min(5, vocab_size - 1)
        input_ids.fill_(token_id)
    elif pattern == "alternating":
        a = 1 if vocab_size <= 2 else min(5, vocab_size - 1)
        b = 1 if vocab_size <= 3 else min(6, vocab_size - 1)
        row = torch.tensor(([a, b] * ((length // 2) + 1))[:length], dtype=torch.long)
        input_ids = row.unsqueeze(0).repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids)
    elif pattern == "explicit_position_ids":
        pad_len = max(1, length // 4) if length > 1 else 0
        if pad_len:
            input_ids[:, :pad_len] = pad_token_id
            attention_mask[:, :pad_len] = 0
    elif pattern == "sliding_window_boundary":
        pass
    elif pattern == "no_padding":
        pass
    else:
        raise ValueError(f"Unsupported long-context pattern: {pattern}")

    batch = {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "labels": _labels_from_ids(input_ids, attention_mask).long(),
    }
    if pattern == "explicit_position_ids":
        batch["position_ids"] = make_position_ids_from_attention_mask(attention_mask).long()
    return batch


def make_long_context_bundles(
    tokenizer: Any | None,
    config: Any | None,
    spec: LongContextSpec | None = None,
) -> list[LongContextBundle]:
    spec = spec or LongContextSpec()
    vocab_size = infer_vocab_size(tokenizer, config)
    pad_token_id = infer_pad_token_id(tokenizer, None) if tokenizer is not None else 0
    if tokenizer is None:
        pad_token_id = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    lengths = infer_context_lengths(config, spec)
    bundles: list[LongContextBundle] = []
    for length in lengths:
        for pattern in spec.patterns:
            if pattern == "sliding_window_boundary" and _get_config_int(config, ("sliding_window",)) is None:
                continue
            batch = _make_pattern_batch(
                pattern,
                length=length,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
                batch_size=spec.batch_size,
                generator=generator,
            )
            bundles.append(
                LongContextBundle(
                    name=f"long.{pattern}.seq{length}",
                    length=length,
                    pattern=pattern,
                    batch=batch,
                    metadata={"vocab_size": vocab_size, "pad_token_id": pad_token_id},
                )
            )
    return bundles


def _select_last_nonpad(logits: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return logits[:, -1, :]
    lengths = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, lengths, :]


def _select_positions(logits: torch.Tensor, positions: tuple[int, ...]) -> torch.Tensor:
    if not positions:
        return logits[:, -1:, :]
    resolved = []
    length = logits.shape[1]
    for pos in positions:
        value = length + pos if pos < 0 else pos
        if 0 <= value < length:
            resolved.append(value)
    if not resolved:
        resolved = [length - 1]
    return logits[:, resolved, :]


def _compare_selected_logits(
    source_logits: torch.Tensor,
    destination_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    bundle: LongContextBundle,
    spec: LongContextSpec,
) -> ValidationReport:
    report = ValidationReport(metrics={"length": bundle.length, "pattern": bundle.pattern})
    selections: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    if bundle.length <= spec.compare_all_logits_below_length:
        selections.append(("all", source_logits, destination_logits))
    if spec.compare_last_token_logits:
        selections.append(("last", source_logits[:, -1, :], destination_logits[:, -1, :]))
    if spec.compare_nonpad_logits:
        selections.append(("nonpad_last", _select_last_nonpad(source_logits, attention_mask), _select_last_nonpad(destination_logits, attention_mask)))
    if spec.selected_positions:
        selections.append(("selected_positions", _select_positions(source_logits, spec.selected_positions), _select_positions(destination_logits, spec.selected_positions)))

    for name, src, dst in selections:
        diff = tensor_diff(src, dst)
        report.metrics[name] = {
            "max_abs": diff.max_abs,
            "mean_abs": diff.mean_abs,
            "rel_l2": diff.rel_l2,
            "cosine": diff.cosine,
        }
        if diff.max_abs > spec.atol and diff.rel_l2 > spec.rtol:
            report.fail(f"{bundle.name}.{name}: logits mismatch max_abs={diff.max_abs:.3g} rel_l2={diff.rel_l2:.3g}")
    return report


@torch.inference_mode()
def validate_long_context_bundle(
    source_model: Any,
    destination_model: Any,
    bundle: LongContextBundle,
    base_spec: ValidationSpec,
    spec: LongContextSpec | None = None,
) -> ValidationReport:
    spec = spec or LongContextSpec()
    report = ValidationReport(metrics={"bundle": bundle.name})
    batch = _move_batch(bundle.batch, base_spec.device)
    source_model.eval().to(base_spec.device)
    destination_model.eval().to(base_spec.device)

    src_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch.get("attention_mask"),
        "position_ids": batch.get("position_ids"),
        "labels": batch.get("labels"),
        "use_cache": False,
        "return_dict": True,
        "output_hidden_states": False,
        "output_attentions": False,
    }
    dst_kwargs = dict(src_kwargs)
    src_kwargs = {key: value for key, value in src_kwargs.items() if value is not None}
    dst_kwargs = {key: value for key, value in dst_kwargs.items() if value is not None}

    if "position_ids" in batch:
        if "position_ids" not in _accepted_kwargs(source_model.forward) or "position_ids" not in _accepted_kwargs(destination_model.forward):
            message = "explicit position_ids requested but source or destination forward does not accept position_ids"
            if spec.require_position_ids_support:
                report.fail(message)
            else:
                report.warnings.append(message)
            src_kwargs.pop("position_ids", None)
            dst_kwargs.pop("position_ids", None)

    src_out = source_model(**_filter_forward_kwargs(source_model, src_kwargs))
    dst_out = destination_model(**_filter_forward_kwargs(destination_model, dst_kwargs))
    src_logits = _get_output_attr(src_out, "logits")
    dst_logits = _get_output_attr(dst_out, "logits")
    if src_logits is None or dst_logits is None:
        report.fail("long-context forward requires logits from both models")
        return report
    child = _compare_selected_logits(src_logits, dst_logits, batch.get("attention_mask"), bundle, spec)
    _merge_child(report, "logits", child)
    return report


def run_long_context_check(
    source_model: Any,
    destination_model: Any,
    *,
    tokenizer: Any | None = None,
    base_spec: ValidationSpec | None = None,
    long_spec: LongContextSpec | None = None,
    generation_spec: GenerationCacheSpec | None = None,
) -> ValidationReport:
    base_spec = base_spec or ValidationSpec(compare_hidden_states=False)
    long_spec = long_spec or LongContextSpec()
    report = ValidationReport(metrics={"enabled": long_spec.enabled})
    if not long_spec.enabled:
        report.warnings.append("long-context check disabled")
        return report
    config = getattr(source_model, "config", None)
    try:
        bundles = make_long_context_bundles(tokenizer, config, long_spec)
    except Exception as exc:
        report.fail(f"failed to construct long-context bundles: {type(exc).__name__}: {exc}")
        return report
    report.metrics["num_bundles"] = len(bundles)
    report.metrics["bundle_names"] = [bundle.name for bundle in bundles]

    for bundle in bundles:
        child = validate_long_context_bundle(source_model, destination_model, bundle, base_spec, long_spec)
        _merge_child(report, bundle.name, child)

    if long_spec.run_generation_cache_at_long_context and generation_spec is not None:
        generation_spec = GenerationCacheSpec(**{**generation_spec.__dict__, "max_new_tokens": long_spec.generation_max_new_tokens})
        candidate_batches = [bundle.batch for bundle in bundles if bundle.pattern in {"left_padding", "explicit_position_ids", "sliding_window_boundary", "no_padding"}]
        candidate_batches = candidate_batches[: long_spec.max_generation_bundles]
        if candidate_batches:
            child = run_generation_cache_check(source_model, destination_model, candidate_batches, base_spec, generation_spec)
            _merge_child(report, "generation_at_long_context", child)
    return report
