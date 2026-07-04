#!/usr/bin/env python3
"""Clone and P0-validate tiny random checkpoints."""
from __future__ import annotations

import argparse, gc, json, shutil, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hypercloning import cloneModel
from hc_validate.random_models import save_pretrained_compat
from hc_validate.run_all import build_all_validation_spec, infer_validation_spec, run_all_validations


def model_dirs(root: Path, names: list[str] | None) -> list[Path]:
    if names: return [root / name for name in names]
    manifest = root / "tiny_random_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        found = [Path(item["path"]) for item in data.get("results", []) if item.get("status") == "created" and item.get("path")]
        if found: return found
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists())


def load_model(path: Path, trust_remote_code: bool, local_files_only: bool):
    return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=local_files_only, torch_dtype=torch.float32)


def load_tokenizer(path: Path, trust_remote_code: bool, local_files_only: bool):
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code, local_files_only=local_files_only)
    except Exception:
        return None


def copy_code_files(src: Path, dst: Path) -> None:
    for pattern in ("configuration_*.py", "modeling_*.py", "tokenization_*.py", "processing_*.py"):
        for file in src.glob(pattern): shutil.copy2(file, dst / file.name)


def compare_logits(src, dst, seq_len: int, atol: float) -> dict[str, float]:
    src.eval(); dst.eval()
    vocab = int(src.get_input_embeddings().weight.shape[0])
    ids = torch.randint(1, max(vocab, 2), (1, seq_len), dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        a = src(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float().cpu()
        b = dst(input_ids=ids, attention_mask=mask, use_cache=False).logits.detach().float().cpu()
    diff = (a - b).abs(); max_abs = float(diff.max().item()); mean_abs = float(diff.mean().item())
    if max_abs > atol: raise RuntimeError(f"logits mismatch max_abs={max_abs:.6g} > {atol:.6g}")
    return {"max_abs": max_abs, "mean_abs": mean_abs}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clone and P0-validate tiny random HyperCloning source checkpoints.")
    p.add_argument("--models-dir", type=Path, default=Path("models/tiny_random")); p.add_argument("--model-names", nargs="+", default=None)
    p.add_argument("--clone-dir", type=Path, default=Path("models/tiny_random_clones")); p.add_argument("--reports-dir", type=Path, default=Path("reports/tiny_random_validation"))
    p.add_argument("--embedding-dim-multiplier", type=int, default=2); p.add_argument("--up-project-multiplier", type=int, default=2); p.add_argument("--num-heads-multiplier", type=int, default=2)
    p.add_argument("--trust-remote-code", action="store_true"); p.add_argument("--local-files-only", action="store_true"); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--p0-seq-lengths", default="8,16"); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--num-batches", type=int, default=1)
    p.add_argument("--logits-seq-len", type=int, default=8); p.add_argument("--logits-atol", type=float, default=5e-3); p.add_argument("--max-shard-size", default="2GB")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv); args.clone_dir.mkdir(parents=True, exist_ok=True); args.reports_dir.mkdir(parents=True, exist_ok=True)
    failures, records = 0, []
    for src_dir in model_dirs(args.models_dir, args.model_names):
        clone_dir = args.clone_dir / f"{src_dir.name}_hc_{args.embedding_dim_multiplier}x{args.up_project_multiplier}"
        report_dir = args.reports_dir / src_dir.name
        if clone_dir.exists():
            if args.overwrite: shutil.rmtree(clone_dir)
            else: print(f"[skip] {clone_dir} exists"); continue
        try:
            print(f"[clone] {src_dir.name}")
            src = load_model(src_dir, args.trust_remote_code, args.local_files_only).eval()
            dst = cloneModel(src, embedding_dim_multiplier=args.embedding_dim_multiplier, up_project_multiplier=args.up_project_multiplier, num_heads_multiplier=args.num_heads_multiplier).eval()
            diff = compare_logits(src, dst, args.logits_seq_len, args.logits_atol)
            clone_dir.mkdir(parents=True, exist_ok=True); save_pretrained_compat(dst, clone_dir, True, args.max_shard_size); copy_code_files(src_dir, clone_dir)
            tokenizer = load_tokenizer(src_dir, args.trust_remote_code, args.local_files_only)
            base, warnings = infer_validation_spec(src.config, dst.config, seq_len=8, batch_size=args.batch_size, num_batches=args.num_batches, embedding_dim_multiplier=args.embedding_dim_multiplier, up_project_multiplier=args.up_project_multiplier, num_heads_multiplier=args.num_heads_multiplier)
            spec = build_all_validation_spec(base, output_dir=report_dir, trust_remote_code=args.trust_remote_code, p0_seq_lengths=tuple(int(x) for x in args.p0_seq_lengths.split(",") if x), run_p1=False, run_save_reload=False, metadata={"source_model": str(src_dir), "cloned_model": str(clone_dir), "inference_warnings": warnings})
            result = run_all_validations(src, dst, tokenizer=tokenizer, spec=spec)
            if not result.summary.get("ok"): raise RuntimeError(f"validation failed: {result.summary}")
            records.append({"name": src_dir.name, "source": str(src_dir), "cloned": str(clone_dir), "report": str(report_dir), "logits": diff})
            del dst; del src
        except Exception as exc:
            failures += 1; records.append({"name": src_dir.name, "status": "failed", "error": str(exc)}); print(f"[failed] {src_dir.name}: {exc}", file=sys.stderr)
        finally:
            gc.collect()
    manifest = {"failures": failures, "records": records}; out = args.reports_dir / "tiny_random_validation_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"); print(f"manifest: {out}")
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
