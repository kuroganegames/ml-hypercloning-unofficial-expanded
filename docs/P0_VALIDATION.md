# P0 HyperCloning validation checks

P0 checks are intended to run early while developing HyperCloning support for a new Hugging Face-style architecture. They keep the validator architecture-independent and focus on clone-time function preservation.

## Included checks

P0 currently includes:

1. **Input matrix providers** in `hc_validate.inputs`
   - random token batches
   - high-token-id batches
   - special-token batches
   - tokenizer fixture batches
   - left- and right-padding variants
   - optional local text file batches
   - optional chat-template batches

2. **Save/reload validation** in `hc_validate.serialization`
   - `destination.save_pretrained(...)`
   - reload through `AutoModelForCausalLM.from_pretrained(...)` or a custom loader
   - compare destination-before-save vs destination-after-reload
   - compare source vs destination-after-reload
   - compare state_dict keys/shapes and config/generation_config metadata

3. **Checkpoint tier runner** in `hc_validate.checkpoint_matrix`
   - tiny config or model-factory cases for CI
   - real checkpoint cases gated by an environment variable such as `HC_VALIDATE_REAL_CHECKPOINTS=1`
   - same input matrix and save/reload checks per case

4. **Suite orchestration** in `hc_validate.suite`
   - run config audit, input-matrix forward comparison, and save/reload on an already cloned source/destination pair

## Example: already cloned pair

```python
from hc_validate import ValidationSpec
from hc_validate.inputs import InputMatrixSpec
from hc_validate.serialization import SaveReloadSpec
from hc_validate.suite import P0ValidationSpec, run_p0_validation

report = run_p0_validation(
    source_model,
    destination_model,
    tokenizer=tokenizer,
    spec=P0ValidationSpec(
        base=ValidationSpec(
            embedding_dim_multiplier=2,
            up_project_multiplier=2,
            num_heads_multiplier=2,
            seq_len=128,
            batch_size=2,
            num_batches=4,
            device="cuda",
        ),
        inputs=InputMatrixSpec(seq_lengths=(8, 32, 128)),
        save_reload=SaveReloadSpec(trust_remote_code=True),
    ),
)

assert report.ok, report.failures
```

## Example: checkpoint matrix

```python
from hc_validate import ValidationSpec
from hc_validate.checkpoint_matrix import CheckpointCase, CheckpointMatrixSpec, run_checkpoint_matrix

matrix = CheckpointMatrixSpec(
    cases=(
        CheckpointCase(
            name="newarch-real",
            source="./checkpoints/newarch-small",
            tokenizer="./checkpoints/newarch-small",
            clone_entry="hypercloning.newarch_cloning:clone_newarch",
            trust_remote_code=True,
            device="cuda",
            requires_env="HC_VALIDATE_REAL_CHECKPOINTS",
        ),
    ),
)

report = run_checkpoint_matrix(
    matrix,
    ValidationSpec(
        embedding_dim_multiplier=2,
        up_project_multiplier=2,
        num_heads_multiplier=2,
        device="cuda",
    ),
)
```

## What P0 does not prove

P0 confirms that the cloned model is function-preserving under the tested input and serialization conditions. It does not prove that the cloned model trains faster than a random baseline, nor that cache-generation, long-context RoPE, or low-precision kernels are correct. Those remain P1/P2 checks.
