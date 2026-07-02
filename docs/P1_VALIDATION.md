# P1 HyperCloning validation checks

P1 extends the P0 function-preservation checks into two paths that often fail even when ordinary forward comparison passes:

1. `use_cache=True` / `generate()` / manual cache decoding
2. long-context position, padding, and attention-mask scenarios

The validator remains architecture-independent. It calls public Hugging Face-style surfaces such as `forward()`, optional `generate()`, `config`, `attention_mask`, `position_ids`, and `past_key_values`. It does not inspect model-specific module names such as `layers[i].self_attn.q_proj`.

## Generation/cache validation

`hc_validate.generation` provides:

- `GenerationCacheSpec`
- `run_generate_compare(...)`
- `manual_cache_loop(...)`
- `run_manual_cache_compare(...)`
- `run_generation_cache_check(...)`

The manual cache loop is the most diagnostic check. It performs prefill with `use_cache=True`, then feeds one token at a time using `past_key_values`. Destination decoding is teacher-forced with source tokens so that a small argmax difference does not cause diverging prefixes.

It checks:

- source vs destination prefill logits
- source vs destination decode-step logits
- optional `generate()` sequence equality
- optional raw generate logits or scores equality
- source no-cache full forward vs source cache loop
- destination no-cache full forward vs destination cache loop

Example:

```python
from hc_validate import ValidationSpec
from hc_validate.generation import GenerationCacheSpec, run_generation_cache_check
from hc_validate.inputs import InputMatrixSpec, make_input_matrix

bundles = make_input_matrix(
    tokenizer,
    source_model.config,
    InputMatrixSpec(seq_lengths=(32,), num_random_batches_per_length=1),
)
batches = [batch for bundle in bundles for batch in bundle.batches]

report = run_generation_cache_check(
    source_model,
    destination_model,
    batches,
    ValidationSpec(
        embedding_dim_multiplier=2,
        up_project_multiplier=2,
        num_heads_multiplier=2,
        device="cuda",
        compare_hidden_states=False,
    ),
    GenerationCacheSpec(max_new_tokens=8, require_past_key_values=False),
)

assert report.ok, report.failures
```

## Long-context validation

`hc_validate.long_context` provides:

- `LongContextSpec`
- `infer_context_lengths(...)`
- `make_position_ids_from_attention_mask(...)`
- `make_long_context_bundles(...)`
- `validate_long_context_bundle(...)`
- `run_long_context_check(...)`

It generates systematic input patterns:

- no padding
- right padding
- left padding
- ragged batch lengths
- repeated same-token prompts
- alternating-token prompts
- explicit `position_ids`
- optional sliding-window boundary cases when `config.sliding_window` exists

For short contexts, it can compare all logits. For long contexts, it compares selected logits such as last-token and non-pad-last logits to avoid excessive memory use.

Example:

```python
from hc_validate import ValidationSpec
from hc_validate.generation import GenerationCacheSpec
from hc_validate.long_context import LongContextSpec, run_long_context_check

report = run_long_context_check(
    source_model,
    destination_model,
    tokenizer=tokenizer,
    base_spec=ValidationSpec(
        embedding_dim_multiplier=2,
        up_project_multiplier=2,
        num_heads_multiplier=2,
        device="cuda",
        compare_hidden_states=False,
    ),
    long_spec=LongContextSpec(
        max_length_cap=2048,
        patterns=(
            "no_padding",
            "left_padding",
            "right_padding",
            "ragged",
            "explicit_position_ids",
        ),
    ),
    generation_spec=GenerationCacheSpec(max_new_tokens=4, run_generate=False),
)
```

## P1 suite

`hc_validate.p1_suite` provides a convenience wrapper:

```python
from hc_validate import ValidationSpec
from hc_validate.generation import GenerationCacheSpec
from hc_validate.inputs import InputMatrixSpec
from hc_validate.long_context import LongContextSpec
from hc_validate.p1_suite import P1ValidationSpec, run_p1_validation

report = run_p1_validation(
    source_model,
    destination_model,
    tokenizer=tokenizer,
    spec=P1ValidationSpec(
        base=ValidationSpec(
            embedding_dim_multiplier=2,
            up_project_multiplier=2,
            num_heads_multiplier=2,
            device="cuda",
            compare_hidden_states=False,
        ),
        prompt_inputs=InputMatrixSpec(seq_lengths=(8, 32, 128)),
        generation=GenerationCacheSpec(max_new_tokens=8),
        long_context=LongContextSpec(max_length_cap=2048),
    ),
)

assert report.ok, report.failures
```

## Common failure hints

| Failure pattern | Likely cause |
| --- | --- |
| P0 forward passes but `generate()` sequence differs | `generation_config`, EOS/PAD handling, logits processors, or small argmax flip |
| Prefill logits pass but decode-step logits fail | `past_key_values`, `cache_position`, `position_ids`, RoPE offset, or KV-head handling |
| Source no-cache vs source cache fails | The source/new architecture cache path is internally inconsistent |
| Destination no-cache vs destination cache fails | Clone-time handling of position, QKV, KV heads, or cache update may be wrong |
| Short context passes but long context fails | RoPE scaling, max-position handling, learned positional embedding clone, or sliding-window mask |
| No-padding passes but left-padding fails | Position IDs derived from `attention_mask` are likely wrong |
| Explicit `position_ids` fails | `forward()` may ignore or mishandle externally supplied position IDs |

## Scope

P1 still focuses on clone-time function preservation. It does not replace low-precision kernel validation, FlashAttention/SDPA backend matrices, training convergence tests, or random-baseline speed comparisons.
