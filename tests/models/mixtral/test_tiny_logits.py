import pytest
import torch


transformers = pytest.importorskip("transformers")


def _mixtral_classes():
    MixtralConfig = getattr(transformers, "MixtralConfig", None)
    MixtralForCausalLM = getattr(transformers, "MixtralForCausalLM", None)
    if MixtralConfig is None or MixtralForCausalLM is None:
        pytest.skip("transformers does not expose Mixtral classes")
    return MixtralConfig, MixtralForCausalLM


def _make_tiny_config():
    MixtralConfig, _ = _mixtral_classes()
    return MixtralConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        router_jitter_noise=0.0,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        output_router_logits=False,
        sliding_window=None,
    )


def _force_eager(model):
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"


def test_mixtral_clone_preserves_tiny_logits_fp32():
    _, MixtralForCausalLM = _mixtral_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)
    config = _make_tiny_config()
    src = MixtralForCausalLM(config).eval()
    _force_eager(src)

    dst = cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2).eval()
    _force_eager(dst)

    input_ids = torch.randint(1, config.vocab_size, (2, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        src_logits = src(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        dst_logits = dst(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    torch.testing.assert_close(dst_logits, src_logits, rtol=1e-4, atol=1e-4)
