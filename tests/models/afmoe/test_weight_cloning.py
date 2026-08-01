import torch

from .helpers import afmoe_classes, make_tiny_dense_afmoe_config


def _assert_repeated_matrix_blocks(dst, src, row_repeat, column_repeat):
    src_rows, src_columns = src.shape
    assert dst.shape == (row_repeat * src_rows, column_repeat * src_columns)
    for row_index in range(row_repeat):
        for column_index in range(column_repeat):
            block = dst[
                row_index * src_rows : (row_index + 1) * src_rows,
                column_index * src_columns : (column_index + 1) * src_columns,
            ]
            torch.testing.assert_close(block, src / column_repeat)


def _assert_repeated_head_projection(dst, src, src_heads, dst_heads, column_repeat):
    head_dim = src.shape[0] // src_heads
    src_hidden = src.shape[1]
    dst_view = dst.reshape(dst_heads, head_dim, column_repeat, src_hidden)
    src_view = src.reshape(src_heads, head_dim, src_hidden)
    head_repeat = dst_heads // src_heads
    for repeat_index in range(head_repeat):
        head_block = dst_view[
            repeat_index * src_heads : (repeat_index + 1) * src_heads
        ]
        for input_repeat in range(column_repeat):
            torch.testing.assert_close(
                head_block[:, :, input_repeat, :],
                src_view / column_repeat,
            )


def test_afmoe_dense_clone_uses_expected_weight_blocks_and_unties_head():
    _, AfmoeForCausalLM = afmoe_classes()
    from hypercloning import cloneModel

    torch.manual_seed(0)
    config = make_tiny_dense_afmoe_config(num_key_value_heads=2, tie_word_embeddings=True)
    src = AfmoeForCausalLM(config).eval()
    assert src.model.embed_tokens.weight is src.lm_head.weight

    dst = cloneModel(src, embedding_dim_multiplier=2, up_project_multiplier=2).eval()

    assert dst.model.embed_tokens.weight is not dst.lm_head.weight

    source_embedding = src.model.embed_tokens.weight.detach()
    destination_embedding = dst.model.embed_tokens.weight.detach()
    torch.testing.assert_close(destination_embedding[:, : config.hidden_size], source_embedding)
    torch.testing.assert_close(destination_embedding[:, config.hidden_size :], source_embedding)

    source_lm_head = src.lm_head.weight.detach()
    destination_lm_head = dst.lm_head.weight.detach()
    torch.testing.assert_close(destination_lm_head[:, : config.hidden_size], source_lm_head / 2)
    torch.testing.assert_close(destination_lm_head[:, config.hidden_size :], source_lm_head / 2)

    src_layer = src.model.layers[0]
    dst_layer = dst.model.layers[0]

    for norm_name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_mlp_layernorm",
        "post_mlp_layernorm",
    ):
        src_weight = getattr(src_layer, norm_name).weight.detach()
        dst_weight = getattr(dst_layer, norm_name).weight.detach()
        torch.testing.assert_close(dst_weight[: config.hidden_size], src_weight)
        torch.testing.assert_close(dst_weight[config.hidden_size :], src_weight)

    _assert_repeated_head_projection(
        dst_layer.self_attn.q_proj.weight.detach(),
        src_layer.self_attn.q_proj.weight.detach(),
        src_heads=config.num_attention_heads,
        dst_heads=dst.config.num_attention_heads,
        column_repeat=2,
    )
    _assert_repeated_head_projection(
        dst_layer.self_attn.gate_proj.weight.detach(),
        src_layer.self_attn.gate_proj.weight.detach(),
        src_heads=config.num_attention_heads,
        dst_heads=dst.config.num_attention_heads,
        column_repeat=2,
    )
    _assert_repeated_head_projection(
        dst_layer.self_attn.k_proj.weight.detach(),
        src_layer.self_attn.k_proj.weight.detach(),
        src_heads=config.num_key_value_heads,
        dst_heads=dst.config.num_key_value_heads,
        column_repeat=2,
    )

    torch.testing.assert_close(
        dst_layer.self_attn.q_norm.weight,
        src_layer.self_attn.q_norm.weight,
    )
    torch.testing.assert_close(
        dst_layer.self_attn.k_norm.weight,
        src_layer.self_attn.k_norm.weight,
    )

    _assert_repeated_matrix_blocks(
        dst_layer.mlp.gate_proj.weight.detach(),
        src_layer.mlp.gate_proj.weight.detach(),
        row_repeat=2,
        column_repeat=2,
    )
    _assert_repeated_matrix_blocks(
        dst_layer.mlp.up_proj.weight.detach(),
        src_layer.mlp.up_proj.weight.detach(),
        row_repeat=2,
        column_repeat=2,
    )
    _assert_repeated_matrix_blocks(
        dst_layer.mlp.down_proj.weight.detach(),
        src_layer.mlp.down_proj.weight.detach(),
        row_repeat=2,
        column_repeat=2,
    )
