from .helpers import llama_classes, make_tiny_llama3_like_config


def test_llama_cloner_registered():
    _, _ = llama_classes()

    import hypercloning  # noqa: F401 - triggers built-in registration
    from hypercloning.llama_cloning import clone_llama
    from hypercloning.registry import get_cloning_function

    config = make_tiny_llama3_like_config(num_key_value_heads=2)

    assert get_cloning_function(config) is clone_llama
