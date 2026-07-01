"""Save/reload validation for cloned models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from hc_validate import ValidationReport, ValidationSpec, validate_forward

ModelLoader = Callable[[str], Any]


@dataclass
class SaveReloadSpec:
    enabled: bool = True
    safe_serialization: bool = True
    save_tokenizer: bool = True
    compare_state_dict_keys: bool = True
    compare_state_dict_shapes: bool = True
    compare_config: bool = True
    compare_generation_config: bool = True
    trust_remote_code: bool = False
    torch_dtype: Any | None = None
    tmp_dir: str | None = None
    keep_tmp: bool = False
    allow_transformers_fallback: bool = True


def _merge_child(parent: ValidationReport, prefix: str, child: ValidationReport) -> None:
    parent.metrics[prefix] = child.metrics
    parent.warnings.extend(f"{prefix}: {w}" for w in child.warnings)
    for failure in child.failures:
        parent.fail(f"{prefix}: {failure}")


def _save_model(model: Any, path: str, spec: SaveReloadSpec) -> None:
    if not hasattr(model, "save_pretrained"):
        raise TypeError(f"{type(model).__name__} does not implement save_pretrained")
    try:
        model.save_pretrained(path, safe_serialization=spec.safe_serialization)
    except TypeError:
        model.save_pretrained(path)


def _default_model_loader(destination_model: Any, spec: SaveReloadSpec) -> ModelLoader:
    def load(path: str) -> Any:
        kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}
        if spec.torch_dtype is not None:
            kwargs["torch_dtype"] = spec.torch_dtype
        if spec.allow_transformers_fallback:
            try:
                from transformers import AutoModelForCausalLM

                return AutoModelForCausalLM.from_pretrained(path, **kwargs)
            except Exception:
                pass
        if hasattr(destination_model.__class__, "from_pretrained"):
            try:
                return destination_model.__class__.from_pretrained(path, **kwargs)
            except TypeError:
                kwargs.pop("trust_remote_code", None)
                kwargs.pop("torch_dtype", None)
                return destination_model.__class__.from_pretrained(path, **kwargs)
        raise TypeError("No loader available. Pass model_loader=... or install transformers.")

    return load


def _simple_config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        try:
            data = config.to_dict()
        except Exception:
            data = {}
    elif hasattr(config, "__dict__"):
        data = dict(config.__dict__)
    elif isinstance(config, dict):
        data = dict(config)
    else:
        return {}
    simple = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            simple[key] = value
    return simple


def _compare_model_metadata(original: Any, reloaded: Any, spec: SaveReloadSpec) -> ValidationReport:
    report = ValidationReport()
    orig_sd = original.state_dict()
    reload_sd = reloaded.state_dict()
    if spec.compare_state_dict_keys:
        missing = sorted(set(orig_sd) - set(reload_sd))
        unexpected = sorted(set(reload_sd) - set(orig_sd))
        report.metrics["state_dict.missing_keys"] = missing[:50]
        report.metrics["state_dict.unexpected_keys"] = unexpected[:50]
        report.metrics["state_dict.num_missing_keys"] = len(missing)
        report.metrics["state_dict.num_unexpected_keys"] = len(unexpected)
        if missing or unexpected:
            report.fail(f"state_dict key mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    if spec.compare_state_dict_shapes:
        shape_mismatches = []
        for key in sorted(set(orig_sd) & set(reload_sd)):
            if tuple(orig_sd[key].shape) != tuple(reload_sd[key].shape):
                shape_mismatches.append((key, tuple(orig_sd[key].shape), tuple(reload_sd[key].shape)))
        report.metrics["state_dict.shape_mismatches"] = shape_mismatches[:50]
        if shape_mismatches:
            report.fail(f"state_dict shape mismatch count={len(shape_mismatches)}")
    if spec.compare_config:
        orig_cfg = _simple_config_dict(getattr(original, "config", None))
        reload_cfg = _simple_config_dict(getattr(reloaded, "config", None))
        mismatches = []
        for key in sorted(set(orig_cfg) | set(reload_cfg)):
            if orig_cfg.get(key) != reload_cfg.get(key):
                mismatches.append((key, orig_cfg.get(key), reload_cfg.get(key)))
        report.metrics["config.num_mismatches"] = len(mismatches)
        report.metrics["config.mismatches"] = mismatches[:50]
        if mismatches:
            report.warnings.append(f"config differs after reload: {len(mismatches)} fields")
    if spec.compare_generation_config:
        orig_gen = _simple_config_dict(getattr(original, "generation_config", None))
        reload_gen = _simple_config_dict(getattr(reloaded, "generation_config", None))
        if orig_gen or reload_gen:
            mismatches = []
            for key in sorted(set(orig_gen) | set(reload_gen)):
                if orig_gen.get(key) != reload_gen.get(key):
                    mismatches.append((key, orig_gen.get(key), reload_gen.get(key)))
            report.metrics["generation_config.num_mismatches"] = len(mismatches)
            report.metrics["generation_config.mismatches"] = mismatches[:50]
            if mismatches:
                report.warnings.append(f"generation_config differs after reload: {len(mismatches)} fields")
    return report


def run_save_reload_check(
    source_model: Any,
    destination_model: Any,
    batches: list[dict[str, torch.Tensor]],
    validation_spec: ValidationSpec,
    *,
    tokenizer: Any | None = None,
    spec: SaveReloadSpec | None = None,
    model_loader: ModelLoader | None = None,
) -> ValidationReport:
    """Validate that a cloned model survives save_pretrained/from_pretrained.

    The check compares destination-before-save vs destination-after-reload and
    source vs destination-after-reload. The first comparison uses a repeat factor
    of one because both models have the same width.
    """

    spec = spec or SaveReloadSpec()
    report = ValidationReport(metrics={"enabled": spec.enabled})
    if not spec.enabled:
        report.warnings.append("save/reload check disabled")
        return report
    if not batches:
        report.fail("save/reload check requires at least one batch")
        return report

    if spec.tmp_dir:
        tmp_context = None
        tmp_path = Path(spec.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
    else:
        tmp_context = tempfile.TemporaryDirectory(prefix="hc_validate_save_reload_")
        tmp_path = Path(tmp_context.name)

    try:
        _save_model(destination_model, str(tmp_path), spec)
        if tokenizer is not None and spec.save_tokenizer and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(str(tmp_path))
            except Exception as exc:
                report.warnings.append(f"tokenizer save failed: {exc!r}")

        loader = model_loader or _default_model_loader(destination_model, spec)
        reloaded = loader(str(tmp_path))
        if hasattr(reloaded, "to"):
            reloaded.to(validation_spec.device)
        if hasattr(reloaded, "eval"):
            reloaded.eval()
        report.metrics["tmp_dir"] = str(tmp_path) if spec.keep_tmp or spec.tmp_dir else "<temporary>"

        metadata_report = _compare_model_metadata(destination_model, reloaded, spec)
        _merge_child(report, "metadata", metadata_report)

        same_width_spec = replace(validation_spec, embedding_dim_multiplier=1)
        dst_reload_report = validate_forward(destination_model, reloaded, batches, same_width_spec)
        _merge_child(report, "destination_vs_reloaded", dst_reload_report)

        src_reload_report = validate_forward(source_model, reloaded, batches, validation_spec)
        _merge_child(report, "source_vs_reloaded_destination", src_reload_report)
    except Exception as exc:
        report.fail(f"save/reload check raised {type(exc).__name__}: {exc}")
    finally:
        if tmp_context is not None and not spec.keep_tmp:
            tmp_context.cleanup()

    return report
