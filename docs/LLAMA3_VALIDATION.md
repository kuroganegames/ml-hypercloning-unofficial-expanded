# Llama3 validation

This document describes how to validate the existing `clone_llama` implementation
against Llama3 / Llama3.x-style Hugging Face models.

## Scope

The Llama3 compatibility work intentionally reuses the existing Llama cloner.
`LlamaConfig` is already registered to `hypercloning.llama_cloning.clone_llama`,
and the cloner keeps `head_dim` unchanged by expanding the number of attention
heads together with `hidden_size`.

The CI tests in `tests/models/llama/` use tiny synthetic `LlamaConfig`
instances. They do not download gated Meta checkpoints.

## CI coverage

Run the Llama-specific validation tests with:

```bash
pytest tests/models/llama -q
```

The tests cover:

- registry resolution from `LlamaConfig` to `clone_llama`
- MQA, GQA, and MHA tiny config expansion
- rejection of requests that would change `head_dim`
- clone-time logits preservation in fp32
- `save_pretrained()` / `from_pretrained()` preservation for tiny clones

## Manual gated-checkpoint validation

Use a local checkpoint or an accessible Hugging Face model ID. The examples
below default to `meta-llama/Llama-3.2-1B`.

### 1. Clone and save a Llama3 checkpoint

```bash
export LLAMA3_MODEL_ID=meta-llama/Llama-3.2-1B
export LLAMA3_CLONED_OUT=./checkpoints/llama3_hc_2x2

python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hypercloning import cloneModel

model_id = os.environ.get("LLAMA3_MODEL_ID", "meta-llama/Llama-3.2-1B")
out_dir = os.environ.get("LLAMA3_CLONED_OUT", "./checkpoints/llama3_hc_2x2")

source = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

cloned = cloneModel(
    source,
    embedding_dim_multiplier=2,
    up_project_multiplier=2,
)

cloned.save_pretrained(out_dir)
AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

print(f"saved cloned model to {out_dir}")
PY
```

### 2. Run a small P0 validation pass

Start with eager attention and short sequences. This validates ordinary forward
passes without exercising cache or generation paths.

```bash
python -m hc_validate.run_all \
  --source-model "$LLAMA3_MODEL_ID" \
  --cloned-model "$LLAMA3_CLONED_OUT" \
  --tokenizer "$LLAMA3_MODEL_ID" \
  --output-dir ./reports/llama3_hc_2x2_p0 \
  --dtype bf16 \
  --attn-implementation eager \
  --load-profile auto-sharded \
  --embedding-dim-multiplier 2 \
  --up-project-multiplier 2 \
  --num-heads-multiplier 2 \
  --p0-seq-lengths 8,16,32 \
  --batch-size 1 \
  --num-batches 1 \
  --skip-p1 \
  --skip-save-reload
```

### 3. Run P1 cache/generation/long-context validation

After P0 passes, enable the cache, generation, and long-context checks.

```bash
python -m hc_validate.run_all \
  --source-model "$LLAMA3_MODEL_ID" \
  --cloned-model "$LLAMA3_CLONED_OUT" \
  --tokenizer "$LLAMA3_MODEL_ID" \
  --output-dir ./reports/llama3_hc_2x2_p1 \
  --dtype bf16 \
  --attn-implementation eager \
  --load-profile auto-sharded \
  --embedding-dim-multiplier 2 \
  --up-project-multiplier 2 \
  --num-heads-multiplier 2 \
  --p0-seq-lengths 8,16,32 \
  --p1-prompt-seq-lengths 8,16,32 \
  --long-context-max-length-cap 256 \
  --max-new-tokens 4 \
  --batch-size 1 \
  --num-batches 1
```

## Notes

- Keep `head_dim` fixed. Use `num_heads_multiplier=embedding_dim_multiplier`.
- Keep `num_key_value_heads=1` fixed for MQA models. For GQA/MHA models, the
  existing cloner expands non-MQA KV head counts with the hidden-size multiplier.
- Prefer `--attn-implementation eager` while validating a new checkpoint. After
  eager passes, repeat with other backends if needed.
- For tokenizer-backed validation, make sure the source and cloned model use
  the same tokenizer and compatible `pad_token_id` settings.
