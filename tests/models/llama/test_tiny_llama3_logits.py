import pytest
import torch

from .helpers import force_eager, llama_classes, make_random_inputs, make_tiny_llama3_like_config


@pytest.mark.parametrize(
    ("num_key_value_heads", "up_project_multiplier"),
    [
        (1, 1),  # MQA path: KV head count stays fixed.
        (2, 2),  # GQA path: KV head count expands with hidden size.
        (4, 2),  # MHA path: KV head count remains equal to attention heads.
    ],
)
def test_llama3_clone_preserves_tiny_logits_fp32(num_key_value_heads, up_project_multiplier):
    _, LlamaForCausalLM = llama_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)

    config = make_tiny_llama3_like_config(num_key_value_heads=num_key_value_heads, explicit_head_dim=True)
    force_eager(config)
    src = LlamaForCausalLM(config).eval()
    force_eager(src)

    dst = cloneModel(
        src,
        embedding_dim_multiplier=2,
        up_project_multiplier=up_project_multiplier,
    ).eval()
    force_eager(dst)

    inputs = make_random_inputs(config)

    with torch.inference_mode():
        src_logits = src(**inputs, use_cache=False).logits
        dst_logits = dst(**inputs, use_cache=False).logits

    torch.testing.assert_close(dst_logits, src_logits, rtol=1e-4, atol=1e-4)
