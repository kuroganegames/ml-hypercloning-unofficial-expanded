# Dense-only AFMoE HyperCloning validation

This document records the validation contract and the first real-checkpoint
results for the dense-only AFMoE HyperCloning implementation introduced in PR 1.

## Scope

The implementation supports `AfmoeForCausalLM` checkpoints for which every
decoder layer uses the dense MLP path:

- `num_dense_layers == num_hidden_layers`
- no router, routed experts, or shared experts are instantiated
- `mup_enabled=False`
- `head_dim` is preserved while query-head count is expanded
- MQA keeps one KV head; GQA/MHA KV-head counts expand with hidden width
- full/sliding `layer_types` are preserved
- a tied source embedding is converted to an untied destination embedding and
  `lm_head`, because the two tensors require different function-preserving
  scaling rules

Routed/shared-expert AFMoE, muP input scaling, canceling-noise initialization,
quantized projection replacements, and sharded/meta-device source models remain
outside PR 1.

## Meaning of "exact clone"

In this implementation, *exact clone* means that no symmetry-breaking noise is
added (`snr_db=None`) and that the cloning algebra preserves the source function
in exact arithmetic. It does not imply bitwise-identical floating-point results
across devices, dtypes, kernels, or matrix shapes.

The source and widened destination execute different GEMM shapes. Floating-point
reduction order may therefore differ even when their mathematical functions are
equivalent. Validation must separate the canonical high-precision check from
operational low-precision characterization.

## Numerical validation contract

### Canonical CPU FP32

Use eager attention and FP32 on CPU as the canonical function-preservation test.
For the first real checkpoint, logits and repeated hidden-state blocks passed
through sequence length 1024 with:

```text
atol = 2e-5
rtol = 2e-5
```

### CUDA FP32

Use eager attention and FP32 on CUDA as an operational-equivalence test. PR 1
uses the following acceptance criteria:

```text
logits and repeated hidden states: atol = 1e-4, rtol = 2e-5
argmax token agreement: 1.0
all outputs finite: required
```

These thresholds should be treated as a declared contract, not tuned to a
single observed maximum. The tiny-model tests retain the repository's existing
`1e-4` FP32 tolerance.

### CUDA BF16

BF16 is a characterization profile in PR 1, not a strict exact-equality gate.
Record at least:

- maximum and mean absolute logit difference
- relative L2 difference and cosine similarity when available
- argmax token agreement
- finite/non-finite status
- cache and generation behavior

A BF16 result must not be described as bitwise or strict-logit exact merely
because the clone was initialized with `snr_db=None`.

For continued training, prefer cloning from an FP32 training checkpoint when
one is available. If only a BF16 export exists, load it into FP32 before cloning
so the destination master parameters do not incur an additional BF16 rounding
step during construction.

## First real-checkpoint result

The following results were reported for commit
`e9d35c516cac9042b3d593d9369ce87b7ed26df4`, using the locally trained
TinyStories-JA/EN dense-only AFMoE checkpoint described in the project notebook.
The source model used hidden size 128, dense intermediate size 384, four decoder
layers, four query heads, two KV heads, and head dimension 32. The destination
used 2x hidden width and 2x dense-FFN width while retaining layer count and head
dimension.

| Check | Result |
| --- | --- |
| Transformers 5.9.0 AFMoE tests | 13 passed |
| Transformers 5.14.0 AFMoE tests | 13 passed |
| CPU FP32 logits/hidden, sequence lengths 8-1024 | passed at `2e-5 / 2e-5` |
| CUDA FP32 maximum absolute logits difference | `5.24520874e-05` |
| CUDA FP32 mean absolute logits difference | `3.12151026e-06` |
| CUDA FP32 maximum hidden-state difference | `7.34329224e-05` |
| CUDA FP32 argmax agreement | `1.0` |
| Save/reload and destination untie | passed |
| Cache prefill/decode | passed |
| Greedy FP32 generation | exact token match |
| CUDA BF16 maximum absolute logits difference | `0.3515625` |
| CUDA BF16 minimum argmax agreement | `0.980469` |
| CUDA BF16 finite outputs | passed |

The CUDA FP32 measurements satisfy the declared `1e-4 / 2e-5` operational
profile but not the stricter CPU `2e-5 / 2e-5` profile. This is evidence of
operational equivalence under the declared tolerance; it is not, by itself, a
proof that all observed differences arise solely from GEMM reduction order.

## Regression requirement

A full-suite failure is not attributable to the AFMoE PR merely because it is
present on the PR branch. Before merge, run the same test command on the PR base
commit and the PR head in the same environment, then compare failing node IDs.
The merge condition is:

```text
failures present only on the PR head = 0
```

Do not add unrelated `xfail` markers in the AFMoE PR to hide pre-existing Llama
or Mixtral compatibility failures.

## Required checks before merge

```bash
pytest -q tests/models/afmoe
```

Run that command under both supported Transformers versions. Then run the full
suite on both the base and head commits, followed by the real-checkpoint CPU
FP32, CUDA FP32, save/reload, cache, generation, and BF16 characterization
workflows.

PR 1 may be merged when all of the following are true:

- AFMoE-specific tests pass under Transformers 5.9.0 and 5.14.0
- the base/head comparison shows no new regression failure
- canonical CPU FP32 logits and repeated hidden states pass
- CUDA FP32 passes the declared operational profile with 100% argmax agreement
- destination embeddings remain untied after save/reload
- cache prefill/decode and greedy generation pass
- unsupported MoE, muP, head-dimension, and `snr_db` requests are rejected
- BF16 metrics are finite and recorded without being misrepresented as strict
  equality
