#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import Qwen3ForCausalLM

from hypercloning.common import (clone_layer_norm, clone_linear_layer,
                                 clone_matrix, clone_rms_norm, rename_config,
                                 scale_linear_layer, scaledLinear)
from hypercloning.gemma_cloning import clone_gemma_qkv_layer


def clone_qwen3_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer from 'src_layer' into 'dst_layer' for Qwen3.

    Arguments:
        dst_layer: Destination attention layer.
        src_layer: Source (pretrained) attention layer.
        snr_db: signal to noise ratio. Defaults to None
    Returns:
        None.
    """
    clone_gemma_qkv_layer(
        dst_layer.q_proj,
        src_layer.q_proj,
        dst_layer.config.num_attention_heads,
        src_layer.config.num_attention_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.k_proj,
        src_layer.k_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.v_proj,
        src_layer.v_proj,
        dst_layer.config.num_key_value_heads,
        src_layer.config.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)
    # Clone Qwen3-specific q_norm and k_norm
    clone_rms_norm(dst_layer.q_norm, src_layer.q_norm)
    clone_rms_norm(dst_layer.k_norm, src_layer.k_norm)
    return dst_layer


def clone_qwen3(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for the Qwen3 family.

    For arguments description, refer to hypercloning.cloneModel.

    Returns:
        Cloned Qwen3 model instance.
    """
    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert (
        num_heads_multiplier == embedding_dim_multiplier
    ), "head_dim expansion is not supported for Qwen3. The number of heads will \
        be automatically computed based on embedding dimension expansion. Do not \
        pass 'num_heads_multiplier' to 'clone_qwen3'"

    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.tie_word_embeddings = False
    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Make an instance of the destination network:
    dst_network = Qwen3ForCausalLM._from_config(config)

    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
    )

    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        dst_layer.self_attn = clone_qwen3_attention(
            dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db
        )
        clone_linear_layer(
            dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj, snr_db=snr_db
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.down_proj, src_layer.mlp.down_proj, snr_db=snr_db
        )
    clone_rms_norm(dst_network.model.norm, src_network.model.norm)
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network
