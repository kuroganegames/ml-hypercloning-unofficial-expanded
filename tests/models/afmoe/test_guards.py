import pytest

from .helpers import afmoe_classes, make_tiny_dense_afmoe_config


def test_afmoe_dense_cloner_rejects_any_moe_layer():
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel
    from hypercloning.models.afmoe import UnsupportedAfmoeConfigurationError

    config = make_tiny_dense_afmoe_config(num_dense_layers=1)
    src = AfmoeForCausalLM(config).eval()
    assert any(layer.moe_enabled for layer in src.model.layers)

    with pytest.raises(
        UnsupportedAfmoeConfigurationError,
        match="num_dense_layers == num_hidden_layers",
    ):
        cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2)


def test_afmoe_dense_cloner_rejects_mup_input_scaling():
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel
    from hypercloning.models.afmoe import UnsupportedAfmoeConfigurationError

    config = make_tiny_dense_afmoe_config(mup_enabled=True)
    src = AfmoeForCausalLM(config).eval()

    with pytest.raises(
        UnsupportedAfmoeConfigurationError,
        match="mup_enabled=True",
    ):
        cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2)


def test_afmoe_dense_cloner_rejects_snr_noise_in_pr1():
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    config = make_tiny_dense_afmoe_config()
    src = AfmoeForCausalLM(config).eval()

    with pytest.raises(NotImplementedError, match="exact cloning only"):
        cloneModel(
            src,
            embedding_dim_multiplier=2,
            up_project_multiplier=2,
            snr_db=40.0,
        )
