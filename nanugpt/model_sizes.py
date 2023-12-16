

model_sizes = {
    'grokking': {
        'n_layer': 2,
        'n_embd': 128,
        'n_head': 4,
        'context_length': 5,
    },
    'gpt2_117m': {
        'n_layer': 12,
        'n_embd': 768,
        'n_head': 12,
        'context_length': 1024
    },
    'gpt2_345m': {
        'n_layer': 24,
        'n_embd': 1024,
        'n_head': 16,
        'context_length': 1024
    },
    'gpt2_762m': {
        'n_layer': 36,
        'n_embd': 1280,
        'n_head': 20,
        'context_length': 1024
    },
    'gpt2_1542m': {
        'n_layer': 48,
        'n_embd': 1600,
        'n_head': 25,
        'context_length': 1024
    },
    'gpt3_125m': {
        'n_layer': 12,
        'n_embd': 768,
        'n_head': 12,
        'context_length': 2048
    },
    'gpt3_350m': {
        'n_layer': 24,
        'n_embd': 1024,
        'n_head': 16,
        'context_length': 2048
    },
    'gpt3_760m': {
        'n_layer': 24,
        'n_embd': 1536,
        'n_head': 16,
        'context_length': 2048
    },
    'gpt3_1.3b': {
        'n_layer': 24,
        'n_embd': 2048,
        'n_head': 24,
        'context_length': 2048
    },
    'gpt3_2.7b': {
        'n_layer': 32,
        'n_embd': 2560,
        'n_head': 32,
        'context_length': 2048
    },
    'gpt3_6.7b': {
        'n_layer': 32,
        'n_embd': 4096,
        'n_head': 32,
        'context_length': 2048
    },
    'gpt3_13b': {
        'n_layer': 40,
        'n_embd': 5140,
        'n_head': 40,
        'context_length': 2048
    },
    'gpt3_175b': {
        'n_layer': 96,
        'n_embd': 12288,
        'n_head': 96,
        'context_length': 2048
    },
    'llama1_6.7b': {
        'n_layer': 32,
        'n_embd': 4096,
        'n_head': 32,
        'context_length': 2048
    },
    'llama1_13b': {
        'n_layer': 40,
        'n_embd': 5120,
        'n_head': 40,
        'context_length': 2048
    },
    'llama1_34b': {
        'n_layer': 60,
        'n_embd': 6656,
        'n_head': 52,
        'context_length': 2048
    },
    'llama1_65b': {
        'n_layer': 80,
        'n_embd': 8192,
        'n_head': 64,
        'context_length': 2048
    },
    'llama2_6.7b': {
        'n_layer': 32,
        'n_embd': 4096,
        'n_head': 32,
        'context_length': 4096
    },
    'llama2_13b': {
        'n_layer': 40,
        'n_embd': 5120,
        'n_head': 40,
        'context_length': 4096
    },
    'llama2_34b': {
        'n_layer': 60,
        'n_embd': 6656,
        'n_head': 52,
        'context_length': 4096
    },
    'llama2_65b': {
        'n_layer': 80,
        'n_embd': 8192,
        'n_head': 64,
        'context_length': 4096
    },
}