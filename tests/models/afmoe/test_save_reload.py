import torch

from .helpers import afmoe_classes, force_eager, make_random_inputs, make_tiny_dense_afmoe_config


def test_afmoe_dense_clone_save_reload_preserves_logits_and_untied_weights(tmp_path):
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)
    config = make_tiny_dense_afmoe_config(tie_word_embeddings=True)
    force_eager(config)
    src = AfmoeForCausalLM(config).eval()
    force_eager(src)

    dst = cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2).eval()
    force_eager(dst)
    dst.save_pretrained(tmp_path, safe_serialization=True)

    reloaded = AfmoeForCausalLM.from_pretrained(tmp_path).eval()
    force_eager(reloaded)

    assert reloaded.config.tie_word_embeddings is False
    assert reloaded.model.embed_tokens.weight is not reloaded.lm_head.weight

    inputs = make_random_inputs(config)
    with torch.inference_mode():
        src_logits = src(**inputs, use_cache=False).logits
        reloaded_logits = reloaded(**inputs, use_cache=False).logits

    torch.testing.assert_close(reloaded_logits, src_logits, rtol=1e-4, atol=1e-4)
