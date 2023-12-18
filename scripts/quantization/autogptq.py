from lm_eval import evaluator
from lm_eval.models.huggingface import AutoCausalLM

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


#import copy
lm = AutoCausalLM("llama", loaded_model = model, loaded_tokenizer = tokenizer)

results = evaluator.simple_evaluate(
    model=lm,
    tasks = ["hellaswag", "arc_challenge", "winogrande", "arc_easy", "boolq"],
    num_fewshot=0,
    batch_size=100,
    limit=500,
    device="cuda:0"
)

print(results)