# Architecture-independent HyperCloning validation

This document describes the validation helpers added for developing HyperCloning support for new Hugging Face-style model architectures.

The architecture-specific cloner may depend on module paths inside `xxx_modeling.py`, but the validator intentionally does not. It checks public model surfaces: `forward()`, `config`, `state_dict()`, and optional `CloneTrace` metadata.

## Plugin contract

A new cloner can keep the legacy API or return a `CloneResult` with an optional trace.

```python
from hypercloning.clone_api import CloneRequest, CloneResult
from hypercloning.trace import CloneTrace


def clone_newarch(source_model, request: CloneRequest) -> CloneResult:
    trace = CloneTrace(metadata={"architecture": source_model.config.model_type})

    # Architecture-specific HyperCloning implementation goes here.
    destination_model = ...

    trace.add_matrix(
        "model.embed_tokens.weight",
        "model.embed_tokens.weight",
        tuple(src_embed.weight.shape),
        tuple(dst_embed.weight.shape),
        kind="embedding",
        normalize_input_repeat=False,
    )

    return CloneResult(model=destination_model, trace=trace)
```

## Checks

The `hc_validate` helpers are architecture-independent and can be used with Qwen3, Llama-like, Gemma-like, or future custom HF implementations as long as the models expose standard `forward()`, `config`, and `state_dict()` behavior.

Available checks include:

- config multiplier audit for hidden size, FFN size, layer count, heads, and vocab size
- clone-time logits/loss comparison
- repeated hidden-state comparison
- trace-based `state_dict()` audit
- short training smoke test for finite loss and gradients

The most important check is clone-time logits preservation. A HyperCloned destination model should initially compute the same logits as the source model, up to expected numerical precision.
