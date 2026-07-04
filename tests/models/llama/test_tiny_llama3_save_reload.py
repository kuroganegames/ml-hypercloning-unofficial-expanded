import torch

from .helpers import force_eager, llama_classes, make_random_inputs, make_tiny_llama3_like_config


def test_llama3_tiny_clone_save_reload_preserves_logits(tmp_path):
    _, LlamaForCausalLM = llama_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)

    config = make_tiny_llama3_like_config(num_key_value_heads=2, explicit_head_dim=True)
    force_eager(config)
    src = LlamaForCausalLM(config).eval()
    force_eager(src)

    dst = cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2).eval()
    force_eager(dst)

    dst.save_pretrained(tmp_path)
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path).eval()
    force_eager(reloaded)

    inputs = make_random_inputs(config)

    with torch.inference_mode():
        src_logits = src(**inputs, use_cache=False).logits
        reloaded_logits = reloaded(**inputs, use_cache=False).logits

    torch.testing.assert_close(reloaded_logits, src_logits, rtol=1e-4, atol=1e-4)
