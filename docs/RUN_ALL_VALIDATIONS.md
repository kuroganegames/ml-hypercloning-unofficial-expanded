# Run all HyperCloning validations

`hc_validate.run_all` runs the currently available architecture-independent validation stack on an already cloned model pair:

- P0 config/input/save-reload validation
- P1 generation/cache validation
- P1 long-context/position/mask validation
- individual JSON reports
- combined JSON and Markdown summaries

The runner accepts both Hugging Face model IDs and local directories for the source and cloned models.

## Basic usage

```bash
python -m hc_validate.run_all \
  --source-model ./checkpoints/source-small \
  --cloned-model ./checkpoints/cloned-large \
  --tokenizer ./checkpoints/source-small \
  --output-dir ./reports/hypercloning_validation \
  --device cuda \
  --dtype fp32 \
  --trust-remote-code \
  --embedding-dim-multiplier 2 \
  --up-project-multiplier 2 \
  --num-heads-multiplier 2
```

Equivalent wrapper:

```bash
python scripts/run_all_validations.py \
  --source-model ./checkpoints/source-small \
  --cloned-model ./checkpoints/cloned-large \
  --output-dir ./reports/hypercloning_validation
```

`--source-model` and `--cloned-model` can each be either a Hugging Face Hub model ID or a local directory accepted by `AutoModelForCausalLM.from_pretrained(...)`.

## Output files

The output directory contains:

| File | Description |
| --- | --- |
| `metadata.json` | CLI arguments, inferred multipliers, model IDs/paths, dtype/device metadata |
| `p0_report.json` | Full P0 report |
| `p1_report.json` | Full P1 report |
| `all_reports.json` | Combined machine-readable report |
| `summary.json` | Compact pass/fail summary |
| `summary.md` | Human-readable Markdown summary |

The process exits with status code `0` when all enabled top-level suites pass, and `1` otherwise.

## Multiplier inference

If the expansion multipliers are omitted, the runner tries to infer them from the source and cloned configs:

```bash
python -m hc_validate.run_all \
  --source-model Qwen/source-small \
  --cloned-model ./cloned-qwen-large \
  --output-dir ./reports/qwen-validation \
  --trust-remote-code
```

Inference uses common config aliases such as `hidden_size`, `intermediate_size`, and `num_attention_heads`. If a multiplier cannot be inferred exactly, the warning is written to `metadata.json`. For strict reviews, pass the multipliers explicitly.

## Recommended first run

For a newly added architecture, start with small inputs and CPU or a single GPU:

```bash
python -m hc_validate.run_all \
  --source-model ./source-tiny \
  --cloned-model ./cloned-tiny \
  --output-dir ./reports/tiny \
  --device cpu \
  --p0-seq-lengths 8,32 \
  --p1-prompt-seq-lengths 8,32 \
  --long-context-lengths 8,32 \
  --long-context-max-length-cap 32 \
  --max-new-tokens 2 \
  --skip-save-reload
```

Then enable save/reload and longer contexts:

```bash
python -m hc_validate.run_all \
  --source-model ./source-small \
  --cloned-model ./cloned-small \
  --tokenizer ./source-small \
  --output-dir ./reports/small-full \
  --device cuda \
  --dtype bf16 \
  --trust-remote-code \
  --p0-seq-lengths 8,32,128 \
  --p1-prompt-seq-lengths 8,32,128 \
  --long-context-max-length-cap 2048 \
  --max-new-tokens 8
```

## Useful switches

| Option | Purpose |
| --- | --- |
| `--skip-p0` | Skip P0 checks |
| `--skip-p1` | Skip P1 checks |
| `--skip-save-reload` | Skip `save_pretrained` / reload validation |
| `--skip-generate` | Skip `generate()` black-box checks |
| `--skip-manual-cache` | Skip manual `use_cache=True` decode checks |
| `--skip-long-context` | Skip long-context checks |
| `--compare-hidden-states` | Also compare repeated hidden states where models return them |
| `--local-text-file PATH` | Add local text fixtures to tokenizer-backed validation inputs |

## Interpreting failures

The most important success criterion is clone-time function preservation: source and cloned model logits should match under the tested conditions. P0 covers ordinary forward and serialization paths. P1 covers common hidden failure paths such as `past_key_values`, `generate()`, left padding, explicit `position_ids`, and longer contexts.

Common patterns:

| Failure | Likely area to inspect |
| --- | --- |
| P0 fails on random/tokenizer inputs | Core clone scaling, tokenizer/vocab compatibility, `lm_head`, or embedding clone |
| P0 passes but save/reload fails | custom config, wrapper module serialization, tied embeddings, or `auto_map` |
| P1 prefill passes but cache decode fails | `past_key_values`, `cache_position`, RoPE offset, or KV-head cloning |
| P1 short context passes but long context fails | position embedding, RoPE scaling, max-position, or sliding-window mask |
| Left-padding only fails | `attention_mask`-derived position IDs |

## Python API

```python
from hc_validate.run_all import (
    build_all_validation_spec,
    infer_validation_spec,
    run_all_validations,
)

base_spec, warnings = infer_validation_spec(
    source_model.config,
    cloned_model.config,
    device="cuda",
    seq_len=128,
    batch_size=2,
    num_batches=2,
)

spec = build_all_validation_spec(
    base_spec,
    output_dir="./reports/api_run",
    trust_remote_code=True,
    p0_seq_lengths=(8, 32, 128),
    p1_prompt_seq_lengths=(8, 32, 128),
    long_context_max_length_cap=2048,
    max_new_tokens=8,
)

result = run_all_validations(
    source_model,
    cloned_model,
    tokenizer=tokenizer,
    spec=spec,
)

assert result.summary["ok"], result.summary
```

## Scope

This runner validates clone-time function preservation. It does not measure training speedup, random-baseline convergence, low-precision kernel matrices, or downstream benchmark accuracy.
