import pytest
import torch

from .helpers import afmoe_classes, force_eager, make_random_inputs, make_tiny_dense_afmoe_config


@pytest.mark.parametrize(
    ("num_key_value_heads", "up_project_multiplier"),
    [
        (1, 1),  # MQA: keep one KV head.
        (2, 2),  # GQA: expand KV heads with hidden width.
        (4, 2),  # MHA: KV heads stay equal to query heads.
    ],
)
def test_afmoe_dense_clone_preserves_tiny_logits_and_hidden_states_fp32(
    num_key_value_heads,
    up_project_multiplier,
):
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)
    config = make_tiny_dense_afmoe_config(num_key_value_heads=num_key_value_heads)
    force_eager(config)
    src = AfmoeForCausalLM(config).eval()
    force_eager(src)

    embedding_dim_multiplier = 2
    dst = cloneModel(
        src,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
    ).eval()
    force_eager(dst)

    assert all(not layer.moe_enabled for layer in dst.model.layers)
    assert not any(
        token in name
        for name, _ in dst.named_parameters()
        for token in (".router.", ".experts.", ".shared_experts.")
    )

    inputs = make_random_inputs(config, seq_len=12)
    with torch.inference_mode():
        src_output = src(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        dst_output = dst(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    torch.testing.assert_close(
        dst_output.logits,
        src_output.logits,
        rtol=1e-4,
        atol=1e-4,
    )

    assert src_output.hidden_states is not None
    assert dst_output.hidden_states is not None
    assert len(dst_output.hidden_states) == len(src_output.hidden_states)

    for src_hidden, dst_hidden in zip(
        src_output.hidden_states,
        dst_output.hidden_states,
        strict=True,
    ):
        assert dst_hidden.shape[:-1] == src_hidden.shape[:-1]
        assert dst_hidden.shape[-1] == embedding_dim_multiplier * src_hidden.shape[-1]

        repeated_blocks = dst_hidden.reshape(
            *dst_hidden.shape[:-1],
            embedding_dim_multiplier,
            src_hidden.shape[-1],
        )
        for block in repeated_blocks.unbind(dim=-2):
            torch.testing.assert_close(
                block,
                src_hidden,
                rtol=1e-4,
                atol=1e-4,
            )
