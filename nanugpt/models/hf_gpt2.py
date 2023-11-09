from typing import List, Optional

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from nanugpt import utils
from nanugpt import glogging as logging

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# for llama: sample values are at https://huggingface.co/EleutherAI/llemma_7b/blob/main/config.json
def get_model(
              n_layer: int, n_embd: int, n_head: int,
              vocab_size: int, context_length: int,
              n_inner:Optional[int], # None
              activation_function:Optional[str], # None, defaults to 'gelu_new'
              resid_dropout: Optional[float], # defaults to 0.1,  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.dropout for residual in attention
              embed_dropout: Optional[float], #  defaults to 0.1, The dropout ratio for the embeddings
              attn_dropout: Optional[float], # defaults to 0.1, The dropout ratio for the attention
              layer_norm_epsilon: Optional[float], # defaults to 1e-5, The epsilon to use in the layer normalization layers
              initializer_range: Optional[float], # defaults to 0.02, The standard deviation of the truncated_normal_initializer for initializing all weight matrices
              use_cache: Optional[bool], # defaults to True, Whether or not the model should return the last key/values attentions (not used by all models)

              bos_token_id: int, # 1
              eos_token_id: int, # 2
              ):

    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions = context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function=activation_function,
        resid_pdrop=resid_dropout,
        embd_pdrop=embed_dropout,
        attn_pdrop=attn_dropout,
        layer_norm_epsilon=layer_norm_epsilon,
        initializer_range=initializer_range,
        use_cache=use_cache,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
    )

    model = GPT2LMHeadModel(config=model_config)

    return model