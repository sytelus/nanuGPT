import ticktoken


def get_tokenizer(encoding_name:str):
    return ticktoken.get_encoding(encoding_name)

