from typing import List, Optional

from transformers.models.llama import LlamaConfig, LlamaForCausalLM

from nanugpt import utils
from nanugpt import glogging as logging

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# for llama: sample values are at https://huggingface.co/EleutherAI/llemma_7b/blob/main/config.json
def get_model(
                n_layer: int, n_embd: int, n_head: int,
                vocab_size: int, context_length: int,

                use_gqa: bool=False, # False for  < 13B
                rope_theta: Optional[float]=10000.0, # theta for RoPE, 10000.0 but should be 10*contex_len
                enable_flash_attn2: Optional[bool]=True,

                use_cache: Optional[bool]=True, # defaults to True, Whether or not the model should return the last key/values attentions (not used by all models)

                # passed to PretrainedConfig
                tie_word_embeddings: Optional[bool]=False, # kept false for supporting parallelization
                bos_token_id: Optional[int]=1, # 1
                eos_token_id: Optional[int]=2, # 2
                pad_token_id: Optional[int]=None, # 0
                torch_dtype: Optional[str]=None, # 'float16'
              ):

    model_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=compute_intermediate_size(n_embd),
        num_key_value_heads=n_head if not use_gqa else 1,
        max_position_embeddings = context_length,
        use_cache=use_cache, # type: ignore
        rope_theta = rope_theta,

        # passed to PretrainedConfig
        tie_word_embeddings = tie_word_embeddings, # type: ignore
        bos_token_id = bos_token_id, # type: ignore
        eos_token_id = eos_token_id, # type: ignore
        pad_token_id = pad_token_id, # type: ignore
        torch_dtype = torch_dtype,
        model_type = 'llama',
        architectures = ["LlamaForCausalLM"],
    )

    flash_attn_used = enable_flash_attn2 and utils.flash_attn_supported()
    logging.info({
                    'flash_attn_used': flash_attn_used,
                    'flash_attn_supported': utils.flash_attn_supported(),
                    'nvidia_sm': utils. nvidia_sm(),
                  })
    if flash_attn_used:
        model_config._flash_attn_2_enabled = True

    model = LlamaForCausalLM(config=model_config)

    return model