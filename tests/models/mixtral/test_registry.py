import pytest


transformers = pytest.importorskip("transformers")


def test_mixtral_cloner_registered():
    MixtralConfig = getattr(transformers, "MixtralConfig", None)
    if MixtralConfig is None:
        pytest.skip("transformers does not expose MixtralConfig")

    import hypercloning  # noqa: F401 - triggers built-in registration
    from hypercloning.models.mixtral import clone_mixtral
    from hypercloning.registry import get_cloning_function

    config = MixtralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )

    assert get_cloning_function(config) is clone_mixtral
