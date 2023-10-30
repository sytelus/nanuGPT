from typing import List
from gptplay import utils
from gptplay import glogging as logging

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

# for llama: sample values are at https://huggingface.co/EleutherAI/llemma_7b/blob/main/config.json
def get_model(
              config_class_name: str, # 'transformers.models.llama.LlamaConfig'
              model_class_name: str,  # 'transformers.models.llama.LlamaForCausalLM'
              n_layer: int, n_embd: int, n_head: int,
              vocab_size: int, context_length: int,
              use_gqa: bool, # False for  < 13B
              tie_word_embeddings: bool, # kept false for supporting parallelization
              rope_theta: float, # theta for RoPE, 10000.0 but should be 10*contex_len
              bos_token_id: int, # 1
              eos_token_id: int, # 2
              pad_token_id: int, # 0
              use_flash_Attn2: bool,
              torch_dtype: str, # 'float16'

              # below two are probably not needed
              architectures: List[str], # ["LlamaForCausalLM"]
              model_type: str, # 'llama'
              ):

    model_config_cls = utils.import_fn(config_class_name)
    model_config = model_config_cls(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=compute_intermediate_size(n_embd),
        num_key_value_heads=n_head if not use_gqa else 1,
        max_position_embeddings = context_length,
        use_cache=True,
        tie_word_embeddings = tie_word_embeddings,
        rope_theta = rope_theta,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
        pad_token_id = pad_token_id,
        torch_dtype = torch_dtype,
        model_type = model_type,
    )

    flash_attn_used = use_flash_Attn2 and utils.flash_attn_supported()
    logging.info(f'flash_attn_used: flash_attn_used')
    if flash_attn_used:
        model_config._flash_attn_2_enabled = True

    model_cls = utils.import_fn(model_class_name)
    model = model_cls(config=model_config)

    return model