import pytest

from .helpers import afmoe_classes, force_eager, make_tiny_dense_afmoe_config


@pytest.mark.parametrize(
    ("num_key_value_heads", "expected_destination_kv_heads"),
    [(1, 1), (2, 4), (4, 8)],
)
def test_afmoe_dense_config_expansion(num_key_value_heads, expected_destination_kv_heads):
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    config = make_tiny_dense_afmoe_config(num_key_value_heads=num_key_value_heads)
    force_eager(config)
    src = AfmoeForCausalLM(config).eval()
    force_eager(src)

    dst = cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=3).eval()

    assert dst.config.hidden_size == 2 * config.hidden_size
    assert dst.config.intermediate_size == 3 * config.intermediate_size
    assert dst.config.num_attention_heads == 2 * config.num_attention_heads
    assert dst.config.num_key_value_heads == expected_destination_kv_heads
    assert dst.config.head_dim == config.head_dim
    assert dst.config.num_hidden_layers == config.num_hidden_layers
    assert dst.config.num_dense_layers == config.num_dense_layers
    assert dst.config.layer_types == config.layer_types
    assert dst.config.layer_types is not config.layer_types
    assert dst.config.sliding_window == config.sliding_window
    assert dst.config.max_position_embeddings == config.max_position_embeddings
    assert dst.config.mup_enabled is False
    assert dst.config.tie_word_embeddings is False


def test_afmoe_registry_resolves_dense_cloner():
    AfmoeConfig, _ = afmoe_classes()
    from hypercloning import get_cloning_function

    clone_fn = get_cloning_function(AfmoeConfig())
    assert clone_fn.__name__ == "clone_afmoe"


def test_afmoe_rejects_head_dim_expansion_request():
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    config = make_tiny_dense_afmoe_config()
    src = AfmoeForCausalLM(config).eval()

    with pytest.raises(AssertionError, match="head_dim expansion is not supported"):
        cloneModel(
            src,
            embedding_dim_multiplier=2,
            up_project_multiplier=2,
            num_heads_multiplier=1,
        )
