class Tokenizer:
    def __init__(self, prime: int, operations: list[str],
                 eos_token="<|eos|>", eq_token="="):
        self.prime = prime
        self.operations = operations
        self.eos_token = eos_token
        self.eq_token = eq_token

        self.all_tokens = [self.eos_token, self.eq_token] +  \
                    list(sorted(operations) + \
                    list(str(i) for i in range(prime)))
        self.vocab_size = len(self.all_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.all_tokens)}

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, token: str)->int:
        return self.token_to_idx[token]

    def __call__(self, token: str)->int:
        return self.token_to_idx[token]

    def decode(self, idx: int)->str:
        return self.idx_to_token[idx]

    def encode(self, token: str)->int:
        return self.token_to_idx[token]

    def encode_batch(self, tokens: list[str])->list[int]:
        return [self.encode(token) for token in tokens]

    def decode_batch(self, idxs: list[int])->list[str]:
        return [self.decode(idx) for idx in idxs]
