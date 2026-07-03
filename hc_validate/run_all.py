"""Run all architecture-independent HyperCloning validations.

This module provides both a Python API and a CLI entrypoint. It accepts a source
model and an already cloned model, each specified as either a Hugging Face model
ID or a local directory, then runs P0 and P1 validation suites and writes
individual reports plus a summary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hc_validate import ValidationReport, ValidationSpec
from hc_validate.generation import GenerationCacheSpec
from hc_validate.inputs import InputMatrixSpec
from hc_validate.long_context import LongContextSpec
from hc_validate.model_loading import ModelLoadSpec, load_hf_causal_lm, load_json_arg
from hc_validate.p1_suite import P1ValidationSpec, run_p1_validation
from hc_validate.reporting import save_report_bundle, summarize_reports, write_json
from hc_validate.serialization import SaveReloadSpec
from hc_validate.suite import P0ValidationSpec, run_p0_validation


CONFIG_ALIASES = {
    "hidden_size": ("hidden_size", "d_model", "n_embd", "model_dim"),
    "intermediate_size": ("intermediate_size", "ffn_dim", "mlp_hidden_size", "n_inner"),
    "num_attention_heads": ("num_attention_heads", "n_head", "n_heads"),
}


@dataclass
class AllValidationSpec:
    base: ValidationSpec
    p0: P0ValidationSpec
    p1: P1ValidationSpec
    run_p0: bool = True
    run_p1: bool = True
    output_dir: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AllValidationResult:
    reports: dict[str, ValidationReport]
    summary: dict[str, Any]
    output_dir: str | None = None


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def _get_config_value(config: Any | None, logical_key: str) -> int | None:
    if config is None:
        return None
    for name in CONFIG_ALIASES[logical_key]:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    if isinstance(config, dict):
        for name in CONFIG_ALIASES[logical_key]:
            value = config.get(name)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _infer_multiplier(src_config: Any, dst_config: Any, key: str) -> tuple[int | None, str | None]:
    src = _get_config_value(src_config, key)
    dst = _get_config_value(dst_config, key)
    if src is None or dst is None:
        return None, f"could not infer {key}: source={src} destination={dst}"
    if dst % src != 0:
        return None, f"could not infer integer {key} multiplier: source={src} destination={dst}"
    return dst // src, None


def infer_validation_spec(
    source_config: Any,
    cloned_config: Any,
    *,
    device: str = "cpu",
    seq_len: int = 128,
    batch_size: int = 2,
    num_batches: int = 2,
    seed: int = 0,
    embedding_dim_multiplier: int | None = None,
    up_project_multiplier: int | None = None,
    num_heads_multiplier: int | None = None,
    compare_hidden_states: bool = False,
) -> tuple[ValidationSpec, list[str]]:
    """Create a ValidationSpec, inferring multipliers when not supplied."""

    warnings: list[str] = []
    if embedding_dim_multiplier is None:
        embedding_dim_multiplier, warning = _infer_multiplier(source_config, cloned_config, "hidden_size")
        if warning:
            warnings.append(warning)
            embedding_dim_multiplier = 1
    if up_project_multiplier is None:
        up_project_multiplier, warning = _infer_multiplier(source_config, cloned_config, "intermediate_size")
        if warning:
            warnings.append(warning)
            up_project_multiplier = embedding_dim_multiplier
    if num_heads_multiplier is None:
        num_heads_multiplier, warning = _infer_multiplier(source_config, cloned_config, "num_attention_heads")
        if warning:
            warnings.append(warning)
            num_heads_multiplier = embedding_dim_multiplier
    return (
        ValidationSpec(
            seq_len=seq_len,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=seed,
            embedding_dim_multiplier=int(embedding_dim_multiplier),
            up_project_multiplier=int(up_project_multiplier),
            num_heads_multiplier=int(num_heads_multiplier) if num_heads_multiplier is not None else None,
            device=device,
            compare_hidden_states=compare_hidden_states,
        ),
        warnings,
    )


def load_tokenizer_optional(
    tokenizer_id_or_path: str | None,
    *,
    fallback_id_or_path: str | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
    local_files_only: bool = False,
):
    target = tokenizer_id_or_path or fallback_id_or_path
    if not target:
        return None
    try:
        from transformers import AutoTokenizer

        kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
        if revision:
            kwargs["revision"] = revision
        return AutoTokenizer.from_pretrained(target, **kwargs)
    except Exception:
        return None


def build_all_validation_spec(
    base: ValidationSpec,
    *,
    output_dir: str | Path | None = None,
    trust_remote_code: bool = False,
    p0_seq_lengths: tuple[int, ...] = (8, 32, 128),
    p1_prompt_seq_lengths: tuple[int, ...] = (8, 32, 128),
    long_context_lengths: tuple[int, ...] | None = None,
    long_context_max_length_cap: int | None = 2048,
    max_new_tokens: int = 8,
    run_save_reload: bool = True,
    run_generate: bool = True,
    run_manual_cache: bool = True,
    run_long_context: bool = True,
    run_p0: bool = True,
    run_p1: bool = True,
    local_text_file: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AllValidationSpec:
    p0_inputs = InputMatrixSpec(
        seq_lengths=p0_seq_lengths,
        batch_size=base.batch_size,
        num_random_batches_per_length=base.num_batches,
        seed=base.seed,
        local_text_file=local_text_file,
    )
    p0 = P0ValidationSpec(
        base=base,
        inputs=p0_inputs,
        save_reload=SaveReloadSpec(trust_remote_code=trust_remote_code),
        run_save_reload=run_save_reload,
    )

    generation = GenerationCacheSpec(
        max_new_tokens=max_new_tokens,
        run_generate=run_generate,
        run_manual_cache_loop=run_manual_cache,
    )
    p1_inputs = InputMatrixSpec(
        seq_lengths=p1_prompt_seq_lengths,
        batch_size=base.batch_size,
        num_random_batches_per_length=1,
        seed=base.seed,
        local_text_file=local_text_file,
    )
    long_context = LongContextSpec(
        lengths=long_context_lengths,
        max_length_cap=long_context_max_length_cap,
        batch_size=base.batch_size,
        seed=base.seed,
        run_generation_cache_at_long_context=run_generate or run_manual_cache,
    )
    p1 = P1ValidationSpec(
        base=base,
        prompt_inputs=p1_inputs,
        generation=generation,
        long_context=long_context,
        run_generation=run_generate or run_manual_cache,
        run_long_context=run_long_context,
    )
    return AllValidationSpec(base=base, p0=p0, p1=p1, run_p0=run_p0, run_p1=run_p1, output_dir=output_dir, metadata=metadata or {})


def run_all_validations(
    source_model: Any,
    cloned_model: Any,
    *,
    tokenizer: Any | None = None,
    spec: AllValidationSpec,
) -> AllValidationResult:
    reports: dict[str, ValidationReport] = {}
    if spec.run_p0:
        reports["p0"] = run_p0_validation(source_model, cloned_model, tokenizer=tokenizer, spec=spec.p0)
    if spec.run_p1:
        reports["p1"] = run_p1_validation(source_model, cloned_model, tokenizer=tokenizer, spec=spec.p1)
    summary = summarize_reports(reports, metadata=spec.metadata)
    if spec.output_dir is not None:
        summary = save_report_bundle(spec.output_dir, reports, metadata=spec.metadata)
    return AllValidationResult(reports=reports, summary=summary, output_dir=str(spec.output_dir) if spec.output_dir else None)


def _profile_defaults(profile: str, prefix: str) -> dict[str, Any]:
    if profile == "single-gpu":
        return {"device": "cuda"}
    if profile == "auto-sharded":
        return {"device_map": "auto", "low_cpu_mem_usage": True}
    if profile == "cpu-offload":
        return {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "offload_state_dict": True,
            "offload_folder": f"./offload/{prefix}",
        }
    if profile == "cpu":
        return {"device": "cpu"}
    return {}


def build_model_load_spec(args: argparse.Namespace, prefix: str) -> ModelLoadSpec:
    defaults = _profile_defaults(args.load_profile, prefix)
    explicit_device = _none_if_empty(getattr(args, f"{prefix}_device"))
    explicit_device_map = _none_if_empty(getattr(args, f"{prefix}_device_map"))

    device_map = explicit_device_map if explicit_device_map is not None else defaults.get("device_map")
    device = explicit_device if explicit_device is not None else defaults.get("device", args.device)
    if device_map is not None:
        device = None

    low_cpu_mem_usage = bool(defaults.get("low_cpu_mem_usage", False) or getattr(args, f"{prefix}_low_cpu_mem_usage"))
    offload_state_dict = bool(defaults.get("offload_state_dict", False) or getattr(args, f"{prefix}_offload_state_dict"))
    offload_folder = _none_if_empty(getattr(args, f"{prefix}_offload_folder")) or defaults.get("offload_folder")
    max_memory = load_json_arg(getattr(args, f"{prefix}_max_memory_json"))
    extra_kwargs = load_json_arg(getattr(args, f"{prefix}_extra_kwargs_json")) or {}

    return ModelLoadSpec(
        trust_remote_code=args.trust_remote_code,
        revision=getattr(args, f"{prefix}_revision"),
        dtype=args.dtype,
        device=device,
        device_map=device_map,
        max_memory=max_memory,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_folder=offload_folder,
        offload_state_dict=offload_state_dict,
        attn_implementation=_none_if_empty(args.attn_implementation),
        local_files_only=args.local_files_only,
        extra_kwargs=extra_kwargs,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all HyperCloning validation suites on source and cloned models.")
    parser.add_argument("--source-model", required=True, help="HF model ID or local directory for the source model")
    parser.add_argument("--cloned-model", required=True, help="HF model ID or local directory for the already cloned model")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer ID/path. Defaults to --source-model")
    parser.add_argument("--output-dir", required=True, help="Directory where validation reports will be written")
    parser.add_argument("--device", default="cpu", help="Backward-compatible single-device fallback")
    parser.add_argument("--dtype", default="auto", help="auto, fp32, bf16, or fp16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default=None, help="Optional Transformers attention implementation, e.g. eager")
    parser.add_argument("--load-profile", choices=("default", "single-gpu", "auto-sharded", "cpu-offload", "cpu"), default="default")
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--cloned-revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)

    parser.add_argument("--source-device", default=None)
    parser.add_argument("--cloned-device", default=None)
    parser.add_argument("--source-device-map", default=None)
    parser.add_argument("--cloned-device-map", default=None)
    parser.add_argument("--source-max-memory-json", default=None, help="Inline JSON or @path max_memory for source model")
    parser.add_argument("--cloned-max-memory-json", default=None, help="Inline JSON or @path max_memory for cloned model")
    parser.add_argument("--source-low-cpu-mem-usage", action="store_true")
    parser.add_argument("--cloned-low-cpu-mem-usage", action="store_true")
    parser.add_argument("--source-offload-folder", default=None)
    parser.add_argument("--cloned-offload-folder", default=None)
    parser.add_argument("--source-offload-state-dict", action="store_true")
    parser.add_argument("--cloned-offload-state-dict", action="store_true")
    parser.add_argument("--source-extra-kwargs-json", default=None, help="Inline JSON or @path extra kwargs for source from_pretrained")
    parser.add_argument("--cloned-extra-kwargs-json", default=None, help="Inline JSON or @path extra kwargs for cloned from_pretrained")

    parser.add_argument("--embedding-dim-multiplier", type=int, default=None)
    parser.add_argument("--up-project-multiplier", type=int, default=None)
    parser.add_argument("--num-heads-multiplier", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-hidden-states", action="store_true")

    parser.add_argument("--p0-seq-lengths", default="8,32,128")
    parser.add_argument("--p1-prompt-seq-lengths", default="8,32,128")
    parser.add_argument("--long-context-lengths", default=None, help="Comma-separated explicit long-context lengths")
    parser.add_argument("--long-context-max-length-cap", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--local-text-file", default=None)

    parser.add_argument("--skip-p0", action="store_true")
    parser.add_argument("--skip-p1", action="store_true")
    parser.add_argument("--skip-save-reload", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-manual-cache", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    source_load_spec = build_model_load_spec(args, "source")
    cloned_load_spec = build_model_load_spec(args, "cloned")

    source_model = load_hf_causal_lm(args.source_model, source_load_spec)
    cloned_model = load_hf_causal_lm(args.cloned_model, cloned_load_spec)
    tokenizer = load_tokenizer_optional(
        args.tokenizer,
        fallback_id_or_path=args.source_model,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision or args.source_revision,
        local_files_only=args.local_files_only,
    )

    base_spec, inference_warnings = infer_validation_spec(
        getattr(source_model, "config", None),
        getattr(cloned_model, "config", None),
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
        embedding_dim_multiplier=args.embedding_dim_multiplier,
        up_project_multiplier=args.up_project_multiplier,
        num_heads_multiplier=args.num_heads_multiplier,
        compare_hidden_states=args.compare_hidden_states,
    )
    metadata = {
        "source_model": args.source_model,
        "cloned_model": args.cloned_model,
        "tokenizer": args.tokenizer or args.source_model,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "load_profile": args.load_profile,
        "source_load_spec": asdict(source_load_spec),
        "cloned_load_spec": asdict(cloned_load_spec),
        "source_hf_device_map": getattr(source_model, "hf_device_map", None),
        "cloned_hf_device_map": getattr(cloned_model, "hf_device_map", None),
        "embedding_dim_multiplier": base_spec.embedding_dim_multiplier,
        "up_project_multiplier": base_spec.up_project_multiplier,
        "num_heads_multiplier": base_spec.num_heads_multiplier,
        "inference_warnings": inference_warnings,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "metadata.json", metadata)

    spec = build_all_validation_spec(
        base_spec,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
        p0_seq_lengths=_parse_int_tuple(args.p0_seq_lengths) or (8, 32, 128),
        p1_prompt_seq_lengths=_parse_int_tuple(args.p1_prompt_seq_lengths) or (8, 32, 128),
        long_context_lengths=_parse_int_tuple(args.long_context_lengths),
        long_context_max_length_cap=args.long_context_max_length_cap,
        max_new_tokens=args.max_new_tokens,
        run_save_reload=not args.skip_save_reload,
        run_generate=not args.skip_generate,
        run_manual_cache=not args.skip_manual_cache,
        run_long_context=not args.skip_long_context,
        run_p0=not args.skip_p0,
        run_p1=not args.skip_p1,
        local_text_file=args.local_text_file,
        metadata=metadata,
    )
    result = run_all_validations(source_model, cloned_model, tokenizer=tokenizer, spec=spec)
    print(f"Overall result: {'PASS' if result.summary.get('ok') else 'FAIL'}")
    print(f"Summary written to: {Path(args.output_dir) / 'summary.md'}")
    if result.summary.get("failed_checks"):
        print("Failed checks: " + ", ".join(result.summary["failed_checks"]))
    return 0 if result.summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
