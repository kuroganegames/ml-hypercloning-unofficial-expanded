import pytest

from .helpers import force_eager, llama_classes, make_tiny_llama3_like_config


def _clone_tiny_config(*, num_key_value_heads: int, embedding_dim_multiplier: int = 2, up_project_multiplier: int = 2):
    _, LlamaForCausalLM = llama_classes()
    from hypercloning import cloneModel

    config = make_tiny_llama3_like_config(num_key_value_heads=num_key_value_heads, explicit_head_dim=True)
    force_eager(config)
    src = LlamaForCausalLM(config).eval()
    force_eager(src)

    dst = cloneModel(
        src,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
    ).eval()
    force_eager(dst)
    return config, dst.config


def test_llama3_config_clone_keeps_mqa_kv_head_count():
    src_config, dst_config = _clone_tiny_config(num_key_value_heads=1)

    assert dst_config.hidden_size == 2 * src_config.hidden_size
    assert dst_config.intermediate_size == 2 * src_config.intermediate_size
    assert dst_config.num_attention_heads == 2 * src_config.num_attention_heads
    assert dst_config.num_key_value_heads == 1


def test_llama3_config_clone_expands_gqa_kv_head_count():
    src_config, dst_config = _clone_tiny_config(num_key_value_heads=2)

    assert dst_config.hidden_size == 2 * src_config.hidden_size
    assert dst_config.intermediate_size == 2 * src_config.intermediate_size
    assert dst_config.num_attention_heads == 2 * src_config.num_attention_heads
    assert dst_config.num_key_value_heads == 2 * src_config.num_key_value_heads


def test_llama3_config_clone_expands_mha_kv_head_count():
    src_config, dst_config = _clone_tiny_config(num_key_value_heads=4)

    assert src_config.num_key_value_heads == src_config.num_attention_heads
    assert dst_config.num_attention_heads == 2 * src_config.num_attention_heads
    assert dst_config.num_key_value_heads == dst_config.num_attention_heads


def test_llama3_config_clone_preserves_explicit_head_dim_when_supported():
    src_config, dst_config = _clone_tiny_config(num_key_value_heads=2)

    if getattr(src_config, "head_dim", None) is not None:
        assert dst_config.head_dim == src_config.head_dim


def test_llama3_rejects_head_dim_expansion_request():
    _, LlamaForCausalLM = llama_classes()
    from hypercloning import cloneModel

    config = make_tiny_llama3_like_config(num_key_value_heads=2, explicit_head_dim=True)
    force_eager(config)
    src = LlamaForCausalLM(config).eval()
    force_eager(src)

    with pytest.raises(AssertionError, match="head_dim expansion is not supported"):
        cloneModel(
            src,
            embedding_dim_multiplier=2,
            up_project_multiplier=2,
            num_heads_multiplier=1,
        )
