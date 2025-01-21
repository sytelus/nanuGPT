from typing import List, Optional

from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from nanugpt import common
from nanugpt.models.model_with_loss import ModelWithLoss

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# for llama: sample values are at https://huggingface.co/EleutherAI/llemma_7b/blob/main/config.json
def get_model(
                vocab_size: int,
                get_loss: Optional[common.GetLossType],

                n_layer: int, n_embd: int, n_head: int,
                context_length: int,
                n_inner:Optional[int]=None, # inner later of MLP, defaults to 4*n_embd

                activation_function:Optional[str]='gelu_new', # defaults to 'gelu_new'

                resid_dropout: Optional[float]=0.1, # defaults to 0.1,  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.dropout for residual in attention
                embed_dropout: Optional[float]=0.1, #  defaults to 0.1, The dropout ratio for the embeddings
                attn_dropout: Optional[float]=0.1, # defaults to 0.1, The dropout ratio for the attention

                layer_norm_epsilon: Optional[float]=1e-5, # defaults to 1e-5, The epsilon to use in the layer normalization layers
                initializer_range: Optional[float]=0.02, # defaults to 0.02, The standard deviation of the truncated_normal_initializer for initializing all weight matrices

                use_cache: Optional[bool]=True, # defaults to True, Whether or not the model should return the last key/values attentions (not used by all models)

                # passed to PretrainedConfig
                tie_word_embeddings: Optional[bool]=True, # kept false for supporting parallelization
                bos_token_id: Optional[int]=50256, # 1
                eos_token_id: Optional[int]=50256, # 2
                pad_token_id: Optional[int]=None, # 0
                torch_dtype: Optional[str]=None, # 'float16'
              ):

    model_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions = context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function=activation_function, # type: ignore
        resid_pdrop=resid_dropout, # type: ignore
        embd_pdrop=embed_dropout, # type: ignore
        attn_pdrop=attn_dropout, # type: ignore
        layer_norm_epsilon=layer_norm_epsilon, # type: ignore
        initializer_range=initializer_range, # type: ignore
        use_cache=use_cache, # type: ignore

        # passed to PretrainedConfig
        tie_word_embeddings = tie_word_embeddings, # type: ignore
        bos_token_id = bos_token_id, # type: ignore
        eos_token_id = eos_token_id, # type: ignore
        pad_token_id = pad_token_id, # type: ignore
        torch_dtype = torch_dtype,
        model_type = 'gpt2',
        architectures = ["GPT2LMHeadModel"],
    )

    model = GPT2LMHeadModel(config=model_config)

    model_with_loss = ModelWithLoss(model=model, get_loss=get_loss)

    return model_with_loss