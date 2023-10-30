import utils


def get_model(
              n_layer: int, n_embd: int, n_head: int,
              vocab_size: int, context_length: int,
              config_class_name: str, # 'transformers.models.llama.LlamaConfig'
              model_class_name: str,  # 'transformers.models.llama.LlamaForCausalLM'
              use_gqa: bool, # False for  < 13B
              tie_word_embeddings: bool, # kept false for supporting parallelization
              rope_theta: float, # theta for RoPE, 10000.0 but should be 10*contex_len
              bos_token_id: int, # 1
              eos_token_id: int, # 2
              pad_token_id: int, # 0
              ):

    model_config_cls = utils.import_fn(config_class_name)
    model_config = model_config_cls(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        intermediate_size=n_embd * 4,
        num_key_value_heads=n_head if not use_gqa else 1,
        max_position_embeddings = context_length,
        use_cache=True,
        tie_word_embeddings = tie_word_embeddings,
        rope_theta = rope_theta,
        bos_token_id = bos_token_id,
        eos_token_id = eos_token_id,
        pad_token_id = pad_token_id,
    )

    model_cls = utils.import_fn(model_class_name)
    model = model_cls(config=model_config)

    return model