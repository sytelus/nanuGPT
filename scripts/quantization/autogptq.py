file_path = "./local/"

from transformers import LlamaForCausalLM, LlamaConfig

# Configuration dictionary as provided
config_dict = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 5461,
    "max_position_embeddings": 2048,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.34.1",
    "use_cache": True,
    "vocab_size": 32064
}

# Create a LLaMA configuration object from the dictionary
config = LlamaConfig(**config_dict)

# Initialize the LLaMA model with the specified configuration
model = LlamaForCausalLM(config)

model.save_pretrained(file_path)


pretrained_model_dir = file_path (path to llama model)
quantized_model_dir = file_path + "quantilized8/"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf" ,trust_remote_code=True, token="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")
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=8,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)


# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model
model.save_quantized(quantized_model_dir)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)



# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

x = torch.tensor([1, 1, 1]).view(1, 3).to("cuda:0")

model(x)