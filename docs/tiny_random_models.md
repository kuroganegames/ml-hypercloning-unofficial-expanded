# Tiny random checkpoints for HyperCloning validation

This pipeline creates small random CausalLM checkpoints and reuses them for HyperCloning validation. It covers two workflows:

1. **Existing architectures**: create random checkpoints from the installed Transformers/HF implementation, for example Llama, Gemma, Gemma2, Mixtral, OPT, GPTNeoX/Pythia, and OLMo.
2. **New architectures**: point to a local Hugging Face-style implementation directory containing `config.json` plus files such as `configuration_xxx.py` and `modeling_xxx.py`.

The checkpoints are random. They are intended for shape, serialization, `cloneModel(...)`, and validation-runner checks.

## Generate built-in tiny random checkpoints

```bash
python -m hc_validate.random_models \
  --out-dir models/tiny_random \
  --families llama gemma gemma2 mixtral opt pythia olmo \
  --dtype float32 \
  --max-params 100000000 \
  --overwrite \
  --verify-load
```

Equivalent wrapper:

```bash
python scripts/create_tiny_random_models.py \
  --out-dir models/tiny_random \
  --families all \
  --overwrite \
  --verify-load
```

The command writes one subdirectory per model plus:

```text
models/tiny_random/tiny_random_manifest.json
```

Each generated model directory also includes `hypercloning_random_model_metadata.json`.

## Clone and validate generated checkpoints

```bash
python scripts/validate_tiny_random_models.py \
  --models-dir models/tiny_random \
  --clone-dir models/tiny_random_clones \
  --reports-dir reports/tiny_random_validation \
  --embedding-dim-multiplier 2 \
  --up-project-multiplier 2 \
  --num-heads-multiplier 2 \
  --trust-remote-code \
  --local-files-only \
  --overwrite
```

The validation script loads each generated source checkpoint, runs `cloneModel(source, 2, 2)`, compares source/destination logits on random tokens, saves the cloned checkpoint, and runs the architecture-independent P0 validation API.

## Generate a target-compatible source from an existing target config

For built-in architectures, the generator can derive random source configs whose dimensions divide a target config. This is useful when checking that a small source can be expanded into a particular target width.

```bash
python -m hc_validate.random_models \
  --target-config ./target_model/config.json \
  --only-target-compatible \
  --families llama \
  --multipliers 2 4 8 \
  --ffn-multipliers 2 4 8 \
  --out-dir models/target_compatible_random \
  --max-params 100000000 \
  --overwrite
```

## Generate random weights for a new local architecture

A new architecture usually starts with local Hugging Face-style files:

```text
my_arch/
  config.json
  configuration_my_arch.py
  modeling_my_arch.py
```

`config.json` should ideally include:

```json
{
  "auto_map": {
    "AutoConfig": "configuration_my_arch.MyArchConfig",
    "AutoModelForCausalLM": "modeling_my_arch.MyArchForCausalLM"
  }
}
```

If `auto_map` is missing, the generator attempts to infer it from the first `configuration_*.py` class ending in `Config` and the first `modeling_*.py` class ending in `ForCausalLM`.

```bash
python -m hc_validate.random_models \
  --architecture-dir ./my_arch \
  --custom-name my-arch-random-h128 \
  --config-overrides-json '{"vocab_size":8192,"hidden_size":128,"intermediate_size":512,"num_hidden_layers":2,"num_attention_heads":4}' \
  --out-dir models/custom_random \
  --trust-remote-code \
  --local-files-only \
  --overwrite \
  --verify-load
```

The generated checkpoint includes copied `configuration_*.py` and `modeling_*.py` files so it can be reloaded with:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "models/custom_random/my-arch-random-h128",
    trust_remote_code=True,
    local_files_only=True,
)
```

After implementing and registering the matching HyperCloning cloner, the same validation script can clone and P0-validate the custom checkpoint.
