import utils

cls = utils.import_fn('transormers.models.llama.LlamaConfig')

model_config = cls()

print(model_config)