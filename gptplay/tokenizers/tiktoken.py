import tiktoken


def get_tokenizer(encoding_name:str):
    return tiktoken.get_encoding(encoding_name)

