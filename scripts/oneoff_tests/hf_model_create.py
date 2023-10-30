import utils

model_config_cls = utils.import_fn('transformers.models.llama.LlamaConfig')
model_config = model_config_cls()

model_cls = utils.import_fn('transformers.models.llama.LlamaForCausalLM')
model = model_cls(config=model_config)


print(model)


