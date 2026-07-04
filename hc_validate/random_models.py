"""Tiny random checkpoint generation for HyperCloning validation."""
from __future__ import annotations

import argparse, gc, inspect, json, os, random, re, shutil, sys, tempfile, traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

VOCAB_SIZE = 8192
MAX_POS = 256
HIDDEN = 128
LAYERS = 2
HEADS = 4
KV_HEADS = 2
FFN = 512
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SUPPORTED_FAMILIES = ("llama", "gemma", "gemma2", "mixtral", "opt", "pythia", "olmo")
ALIASES = {"all":"all", "llama":"llama", "gemma":"gemma", "gemma2":"gemma2", "gemma-2":"gemma2", "mixtral":"mixtral", "opt":"opt", "pythia":"pythia", "gpt-neox":"pythia", "gpt_neox":"pythia", "gptneox":"pythia", "olmo":"olmo"}

@dataclass
class RandomModelSpec:
    name: str
    family: str
    config: Any
    source: str = "generic"
    notes: str = ""
    architecture_dir: Path | None = None
    trust_remote_code: bool = False
    intended_hypercloning: dict[str, Any] = field(default_factory=dict)


def safe_filename(text: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() or ch in "-_" else "-" for ch in text).strip("-_")
    while "--" in s: s = s.replace("--", "-")
    return s or "model"


def normalize_families(items: Sequence[str]) -> list[str]:
    out = []
    for item in items:
        key = item.lower().strip()
        if key not in ALIASES:
            raise ValueError(f"Unknown family {item!r}; supported: {', '.join(SUPPORTED_FAMILIES)}")
        val = ALIASES[key]
        if val == "all": return list(SUPPORTED_FAMILIES)
        if val not in out: out.append(val)
    return out


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[name.lower()]


def set_global_seed(seed: int) -> None:
    random.seed(seed); set_seed(seed); torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def load_json_arg(value: str | None) -> dict[str, Any] | None:
    if not value: return None
    if value.startswith("@"): return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def apply_json_overrides(config: Any, overrides: dict[str, Any] | None) -> Any:
    for k, v in (overrides or {}).items(): setattr(config, k, v)
    return config


def common(config: Any) -> Any:
    if hasattr(config, "use_cache"): config.use_cache = False
    return config


def parse_class_names(path: Path) -> list[str]:
    try: text = path.read_text(encoding="utf-8")
    except Exception: return []
    return re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[(:]", text, flags=re.MULTILINE)


def infer_auto_map_from_architecture_dir(path: Path) -> dict[str, str]:
    auto_map = {}
    configs = sorted(path.glob("configuration_*.py"))
    models = sorted(path.glob("modeling_*.py"))
    if configs:
        names = [n for n in parse_class_names(configs[0]) if n.endswith("Config")]
        if names: auto_map["AutoConfig"] = f"{configs[0].stem}.{names[0]}"
    if models:
        names = [n for n in parse_class_names(models[0]) if n.endswith("ForCausalLM")]
        if names: auto_map["AutoModelForCausalLM"] = f"{models[0].stem}.{names[0]}"
    return auto_map


def copy_architecture_code_files(src: Path, dst: Path) -> None:
    for pat in ("configuration_*.py", "modeling_*.py", "tokenization_*.py", "processing_*.py", "image_processing_*.py", "feature_extraction_*.py"):
        for path in src.glob(pat): shutil.copy2(path, dst / path.name)


def infer_family(config: Any) -> str:
    mt, cn = str(getattr(config, "model_type", "")).lower(), type(config).__name__.lower()
    if mt == "llama" or "llamaconfig" in cn: return "llama"
    if mt == "gemma2" or "gemma2config" in cn: return "gemma2"
    if mt == "gemma" or "gemmaconfig" in cn: return "gemma"
    if mt == "mixtral" or "mixtralconfig" in cn: return "mixtral"
    if mt == "opt" or "optconfig" in cn: return "opt"
    if mt == "gpt_neox" or "gptneoxconfig" in cn: return "pythia"
    if mt in {"hf_olmo", "olmo"} or "olmoconfig" in cn: return "olmo"
    return mt or "custom"


def int_attr(config: Any, name: str) -> int:
    value = getattr(config, name)
    if value is None: raise ValueError(f"config field {name!r} is None")
    return int(value)


def maybe_divide(value: int, divisor: int) -> int | None:
    return value // divisor if divisor > 0 and value % divisor == 0 else None


def apply_tie_override(config: Any, mode: str) -> None:
    if mode == "keep": return
    value = mode == "true"
    if hasattr(config, "tie_word_embeddings"): config.tie_word_embeddings = value
    if hasattr(config, "weight_tying"): config.weight_tying = value


def build_llama() -> RandomModelSpec:
    from transformers import LlamaConfig
    cfg = LlamaConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, max_position_embeddings=MAX_POS, attention_dropout=0.0, attention_bias=False, mlp_bias=False, tie_word_embeddings=False, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("llama-random-h128-l2-v8k", "llama", common(cfg))


def build_gemma() -> RandomModelSpec:
    from transformers import GemmaConfig
    cfg = GemmaConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HIDDEN//HEADS, max_position_embeddings=MAX_POS, attention_bias=False, attention_dropout=0.0, tie_word_embeddings=True, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("gemma-random-h128-l2-v8k", "gemma", common(cfg))


def build_gemma2() -> RandomModelSpec:
    from transformers import Gemma2Config
    cfg = Gemma2Config(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, head_dim=HIDDEN//HEADS, max_position_embeddings=MAX_POS, attention_bias=False, attention_dropout=0.0, sliding_window=MAX_POS, tie_word_embeddings=True, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("gemma2-random-h128-l2-v8k", "gemma2", common(cfg))


def build_mixtral() -> RandomModelSpec:
    from transformers import MixtralConfig
    cfg = MixtralConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=MAX_POS, router_jitter_noise=0.0, attention_dropout=0.0, sliding_window=None, tie_word_embeddings=False, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("mixtral-random-h128-l2-e4-v8k", "mixtral", common(cfg))


def build_opt() -> RandomModelSpec:
    from transformers import OPTConfig
    cfg = OPTConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, word_embed_proj_dim=HIDDEN, ffn_dim=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, max_position_embeddings=MAX_POS, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, layerdrop=0.0, tie_word_embeddings=True, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("opt-random-h128-l2-v8k", "opt", common(cfg))


def build_pythia() -> RandomModelSpec:
    from transformers import GPTNeoXConfig
    cfg = GPTNeoXConfig(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=LAYERS, num_attention_heads=HEADS, max_position_embeddings=MAX_POS, attention_dropout=0.0, hidden_dropout=0.0, tie_word_embeddings=False, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False)
    return RandomModelSpec("pythia-gptneox-random-h128-l2-v8k", "pythia", common(cfg))


def build_olmo() -> RandomModelSpec:
    from hf_olmo import OLMoConfig
    cfg = OLMoConfig(d_model=HIDDEN, n_heads=HEADS, n_layers=LAYERS, mlp_ratio=4, mlp_hidden_size=FFN, activation_type="swiglu", block_type="sequential", rope=True, alibi=False, max_sequence_length=MAX_POS, vocab_size=VOCAB_SIZE, embedding_size=VOCAB_SIZE, weight_tying=True, include_bias=False, attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0, attention_layer_norm=False, layer_norm_with_affine=False, attention_layer_norm_with_affine=False, bias_for_layer_norm=False, multi_query_attention=False, pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS, use_cache=False, init_device="cpu", init_fn="normal", init_std=0.02)
    return RandomModelSpec("olmo-random-h128-l2-v8k", "olmo", common(cfg), trust_remote_code=True)

BUILDERS: dict[str, Callable[[], RandomModelSpec]] = {"llama": build_llama, "gemma": build_gemma, "gemma2": build_gemma2, "mixtral": build_mixtral, "opt": build_opt, "pythia": build_pythia, "olmo": build_olmo}


def load_custom_config(path: Path, local_files_only: bool) -> Any:
    try:
        return AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=local_files_only)
    except Exception:
        auto_map = infer_auto_map_from_architecture_dir(path)
        if not auto_map:
            raise
        with tempfile.TemporaryDirectory() as tmp:
            patched = Path(tmp) / path.name
            shutil.copytree(path, patched)
            cfg_path = patched / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["auto_map"] = auto_map
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return AutoConfig.from_pretrained(patched, trust_remote_code=True, local_files_only=True)


def load_custom_architecture_spec(path: Path, name: str | None, overrides: dict[str, Any] | None, local_files_only: bool) -> RandomModelSpec:
    cfg = load_custom_config(path, local_files_only)
    apply_json_overrides(cfg, overrides)
    if not getattr(cfg, "auto_map", None):
        auto_map = infer_auto_map_from_architecture_dir(path)
        if auto_map: cfg.auto_map = auto_map
    return RandomModelSpec(name or f"{safe_filename(path.name)}-random", infer_family(cfg), common(cfg), source="custom-architecture-dir", architecture_dir=path, trust_remote_code=True)


def scale_down_target_config(target: Any, family: str, emb_m: int, ffn_m: int, tie_override: str) -> Any | None:
    cfg = json.loads(json.dumps(target.to_dict(), default=str)) if hasattr(target, "to_dict") else None
    cfg = target.__class__(**cfg) if cfg is not None else None
    if cfg is None: return None
    if family in {"llama", "gemma", "gemma2", "mixtral"}:
        h, f, heads = maybe_divide(int_attr(cfg, "hidden_size"), emb_m), maybe_divide(int_attr(cfg, "intermediate_size"), ffn_m), maybe_divide(int_attr(cfg, "num_attention_heads"), emb_m)
        if h is None or f is None or heads is None: return None
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads = h, f, heads
        if hasattr(cfg, "num_key_value_heads") and getattr(cfg, "num_key_value_heads") is not None:
            kv = maybe_divide(int_attr(cfg, "num_key_value_heads"), emb_m)
            if kv is None or (kv == 1 and int_attr(target, "num_key_value_heads") != 1): return None
            cfg.num_key_value_heads = kv
        if hasattr(cfg, "head_dim") and getattr(cfg, "head_dim", None): cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
    elif family == "opt":
        h, f, heads = maybe_divide(int_attr(cfg, "hidden_size"), emb_m), maybe_divide(int_attr(cfg, "ffn_dim"), ffn_m), maybe_divide(int_attr(cfg, "num_attention_heads"), emb_m)
        if h is None or f is None or heads is None: return None
        cfg.hidden_size, cfg.ffn_dim, cfg.num_attention_heads = h, f, heads
        if getattr(cfg, "word_embed_proj_dim", None) == int_attr(target, "hidden_size"): cfg.word_embed_proj_dim = h
    elif family == "pythia":
        h, f, heads = maybe_divide(int_attr(cfg, "hidden_size"), emb_m), maybe_divide(int_attr(cfg, "intermediate_size"), ffn_m), maybe_divide(int_attr(cfg, "num_attention_heads"), emb_m)
        if h is None or f is None or heads is None: return None
        cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads = h, f, heads
    elif family == "olmo":
        h, heads = maybe_divide(int_attr(cfg, "d_model"), emb_m), maybe_divide(int_attr(cfg, "n_heads"), emb_m)
        if h is None or heads is None: return None
        mlp = maybe_divide(int(getattr(cfg, "mlp_hidden_size", int_attr(target, "d_model") * 4)), ffn_m)
        if mlp is None: return None
        cfg.d_model, cfg.n_heads, cfg.mlp_hidden_size = h, heads, mlp
    else:
        return None
    apply_tie_override(cfg, tie_override)
    return common(cfg)


def target_compatible_specs(target_config: Any, target_label: str, families: list[str], multipliers: list[int], ffn_multipliers: list[int], tie_override: str) -> list[RandomModelSpec]:
    family = infer_family(target_config)
    if family not in families: return []
    specs, seen = [], set()
    for em in multipliers:
        for fm in ffn_multipliers:
            cfg = scale_down_target_config(target_config, family, em, fm, tie_override)
            if cfg is None: continue
            key = json.dumps(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__, sort_keys=True, default=str)
            if key in seen: continue
            seen.add(key)
            specs.append(RandomModelSpec(f"{family}-{safe_filename(target_label)}-compatible-x{em}-ffn{fm}", family, cfg, source="target-compatible", intended_hypercloning={"cloneModel_kwargs": {"embedding_dim_multiplier": em, "up_project_multiplier": fm}, "target_config": target_label}, trust_remote_code=(family == "olmo")))
    return specs


def build_specs(args: argparse.Namespace) -> list[RandomModelSpec]:
    families, specs = normalize_families(args.families), []
    if not args.only_target_compatible and not args.only_custom:
        for fam in families:
            try: specs.append(BUILDERS[fam]())
            except Exception as exc:
                if args.strict: raise
                print(f"[skip] generic {fam}: {exc}", file=sys.stderr)
    if args.target_config:
        src = str(args.target_config.parent if Path(args.target_config).exists() and Path(args.target_config).is_file() else args.target_config)
        target = AutoConfig.from_pretrained(src, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only)
        ffn = args.ffn_multipliers or args.multipliers
        specs.extend(target_compatible_specs(target, safe_filename(str(args.target_config)), families, args.multipliers, ffn, args.force_tie_word_embeddings))
    if args.architecture_dir:
        specs.append(load_custom_architecture_spec(args.architecture_dir, args.custom_name, load_json_arg(args.config_overrides_json), args.local_files_only))
    used, out = set(), []
    for spec in specs:
        name, i = spec.name, 2
        while name in used: name, i = f"{spec.name}-{i}", i + 1
        if name != spec.name: spec = RandomModelSpec(name, spec.family, spec.config, spec.source, spec.notes, spec.architecture_dir, spec.trust_remote_code, spec.intended_hypercloning)
        used.add(name); out.append(spec)
    return out


def estimate_params(config: Any, family: str) -> int:
    if family == "olmo":
        h, l, f, v = int_attr(config, "d_model"), int_attr(config, "n_layers"), int(getattr(config, "mlp_hidden_size", HIDDEN * 4)), int(getattr(config, "embedding_size", getattr(config, "vocab_size", VOCAB_SIZE)))
    else:
        h, l, f, v = int(getattr(config, "hidden_size", HIDDEN)), int(getattr(config, "num_hidden_layers", LAYERS)), int(getattr(config, "intermediate_size", getattr(config, "ffn_dim", FFN))), int(getattr(config, "vocab_size", VOCAB_SIZE))
    experts = int(getattr(config, "num_local_experts", 1) or 1)
    return int(v * h * (1 if getattr(config, "tie_word_embeddings", False) else 2) + l * (4 * h * h + experts * 3 * h * f))


def instantiate(config: Any, family: str, dtype: torch.dtype, trust_remote_code: bool) -> torch.nn.Module:
    if family == "olmo":
        try:
            from hf_olmo import OLMoForCausalLM
            try: model = OLMoForCausalLM(config, init_params=True)
            except TypeError: model = OLMoForCausalLM(config)
        except Exception:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    model.eval()
    if dtype is not torch.float32: model.to(dtype=dtype)
    return model


def count_unique_params(model: torch.nn.Module) -> int:
    seen, total = set(), 0
    for p in model.parameters():
        ptr = p.untyped_storage().data_ptr() if hasattr(p, "untyped_storage") else p.data_ptr()
        if ptr not in seen: seen.add(ptr); total += p.numel()
    return int(total)


def save_pretrained_compat(model: torch.nn.Module, out: Path, safe_serialization: bool = True, max_shard_size: str = "2GB") -> None:
    try: model.save_pretrained(out, safe_serialization=safe_serialization, max_shard_size=max_shard_size)
    except TypeError as exc:
        if "safe_serialization" not in str(exc): raise
        model.save_pretrained(out, max_shard_size=max_shard_size)


def write_dummy_tokenizer(out: Path, vocab_size: int, max_length: int) -> None:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
    except Exception:
        print("[warn] tokenizers not installed; dummy tokenizer skipped", file=sys.stderr); return
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK, **{f"tok_{i}": i for i in range(4, vocab_size)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")); backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", model_max_length=max_length).save_pretrained(out)


def patch_olmo(out: Path) -> None:
    (out / "configuration_olmo.py").write_text("from hf_olmo import OLMoConfig\n", encoding="utf-8")
    (out / "modeling_olmo.py").write_text("from hf_olmo import OLMoForCausalLM\n", encoding="utf-8")
    cfg_path = out / "config.json"; cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["auto_map"] = {"AutoConfig": "configuration_olmo.OLMoConfig", "AutoModelForCausalLM": "modeling_olmo.OLMoForCausalLM"}
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def smoke_forward(model: torch.nn.Module, vocab_size: int) -> None:
    ids = torch.randint(1, max(vocab_size, 2), (1, 8), dtype=torch.long, device=next(model.parameters()).device)
    kwargs = {"input_ids": ids}
    try:
        if "use_cache" in inspect.signature(model.forward).parameters: kwargs["use_cache"] = False
    except Exception: pass
    with torch.inference_mode(): out = model(**kwargs)
    if not hasattr(out, "logits") or out.logits.shape[:2] != ids.shape: raise RuntimeError("forward/logits smoke check failed")


def create_one(spec: RandomModelSpec, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    out, tmp = args.out_dir / spec.name, args.out_dir / f".{spec.name}.tmp"
    if out.exists():
        if not args.overwrite: return {"name": spec.name, "family": spec.family, "status": "skipped_exists", "path": str(out)}
        shutil.rmtree(out)
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        est = estimate_params(spec.config, spec.family) if spec.family in SUPPORTED_FAMILIES else None
        if est is not None and est > int(args.max_params * 1.25): return {"name": spec.name, "family": spec.family, "status": "skipped_estimated_too_large", "estimated_parameters": est}
        if args.configs_only:
            spec.config.save_pretrained(tmp); exact = None
        else:
            set_global_seed(seed); model = instantiate(spec.config, spec.family, dtype_from_name(args.dtype), args.trust_remote_code or spec.trust_remote_code)
            exact = count_unique_params(model)
            if exact > args.max_params: raise RuntimeError(f"{exact:,} parameters exceeds max_params={args.max_params:,}")
            vocab = int(getattr(spec.config, "vocab_size", getattr(spec.config, "embedding_size", VOCAB_SIZE)))
            smoke_forward(model, vocab)
            save_pretrained_compat(model, tmp, args.safe_serialization, args.max_shard_size)
        if spec.architecture_dir: copy_architecture_code_files(spec.architecture_dir, tmp)
        if spec.family == "olmo": patch_olmo(tmp)
        if args.tokenizer_source:
            AutoTokenizer.from_pretrained(args.tokenizer_source, trust_remote_code=args.trust_remote_code, local_files_only=args.local_files_only).save_pretrained(tmp)
        elif args.write_dummy_tokenizer:
            write_dummy_tokenizer(tmp, int(getattr(spec.config, "vocab_size", getattr(spec.config, "embedding_size", VOCAB_SIZE))), int(getattr(spec.config, "max_position_embeddings", getattr(spec.config, "max_sequence_length", MAX_POS))))
        meta = {"name": spec.name, "family": spec.family, "source": spec.source, "architecture_dir": str(spec.architecture_dir) if spec.architecture_dir else None, "random_init_seed": seed, "dtype": args.dtype, "unique_parameters": exact, "estimated_parameters": est, "trust_remote_code": spec.trust_remote_code, "intended_hypercloning": spec.intended_hypercloning}
        (tmp / "hypercloning_random_model_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        if args.verify_load and not args.configs_only:
            reloaded = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32); smoke_forward(reloaded, int(getattr(spec.config, "vocab_size", VOCAB_SIZE))); del reloaded
        tmp.rename(out)
        return {"name": spec.name, "family": spec.family, "status": "created", "path": str(out), "unique_parameters": exact, "estimated_parameters": est, "intended_hypercloning": spec.intended_hypercloning}
    finally:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create tiny random CausalLM checkpoints for HyperCloning validation.")
    p.add_argument("--out-dir", type=Path, default=Path("models/tiny_random")); p.add_argument("--max-params", type=int, default=100_000_000)
    p.add_argument("--families", nargs="+", default=["all"]); p.add_argument("--target-config", type=Path, default=None); p.add_argument("--only-target-compatible", action="store_true")
    p.add_argument("--multipliers", nargs="+", type=int, default=[2,4,8]); p.add_argument("--ffn-multipliers", nargs="+", type=int, default=None); p.add_argument("--force-tie-word-embeddings", choices=["keep","true","false"], default="keep")
    p.add_argument("--architecture-dir", type=Path, default=None); p.add_argument("--custom-name", default=None); p.add_argument("--config-overrides-json", default=None); p.add_argument("--only-custom", action="store_true")
    p.add_argument("--dtype", default="float32"); p.add_argument("--seed", type=int, default=1234); p.add_argument("--configs-only", action="store_true"); p.add_argument("--tokenizer-source", default=None); p.add_argument("--trust-remote-code", action="store_true"); p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--write-dummy-tokenizer", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--max-shard-size", default="2GB"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--verify-load", action="store_true"); p.add_argument("--strict", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args)
    if not specs: print("No specs generated", file=sys.stderr); return 2
    results = []
    for i, spec in enumerate(specs):
        try:
            print(f"[create] {spec.name} ({spec.family}, {spec.source})"); result = create_one(spec, args, args.seed + i); results.append(result); print(f"[{result['status']}] {spec.name}: {result.get('path','')}")
        except Exception as exc:
            results.append({"name": spec.name, "family": spec.family, "status": "failed", "error": str(exc)}); print(f"[failed] {spec.name}: {exc}", file=sys.stderr)
            if args.strict: traceback.print_exc(); raise
    manifest = {"max_params": args.max_params, "dtype": args.dtype, "configs_only": args.configs_only, "results": results}
    path = args.out_dir / "tiny_random_manifest.json"; path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"manifest: {path}")
    return 0 if all(r.get("status") != "failed" for r in results) else 1

if __name__ == "__main__":
    raise SystemExit(main())
