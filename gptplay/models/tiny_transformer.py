from einops import rearrange, repeat
import torch
from torch import nn, Tensor

class DecoderBlock(torch.nn.Module):
  def __init__(self, n_embd: int, n_heads: int,
               ffn_bias, attn_proj_bias, attn_kv_bias,
               attn_dropout, ffn_dropout):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(n_embd, n_heads, dropout=attn_dropout,
                                           bias=attn_proj_bias, add_bias_kv=attn_kv_bias)
    self.self_attn_norm = nn.LayerNorm(n_embd)
    self.ffn = nn.Sequential(
        nn.Linear(n_embd, n_embd * 4, bias=ffn_bias),
        nn.GELU(),
        nn.Linear(n_embd * 4, n_embd, bias=ffn_bias),
        nn.Dropout(ffn_dropout)
    )
    self.ffn_norm = nn.LayerNorm(n_embd)

  def forward(self, x: Tensor):
    # x: (context_len, batch_size, n_embd)
    # attn_mask: (context_len, context_len)
    # attn_mask = torch.full(
    #     (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    # )
    # attn_mask = torch.triu(attn_mask, diagonal=1)

    attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[0], device=x.device)

    # a1: (context_len, batch_size, n_embd)
    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    a1 = self.self_attn_norm (x + a1)
    # a1: (context_len, batch_size, n_embd)
    a2 = self.ffn(a1)
    a2 = self.ffn_norm(a1 + a2)

    return a2

class TinyTransformer(torch.nn.Module):
    def __init__(self, n_layer: int, n_embd: int, n_head: int,
                vocab_size: int, context_len: int,
                mlp_bias: bool, attn_proj_bias: float, attn_kv_bias: float,
                attn_dropout: float, mlp_dropout: float):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(context_len, n_embd)
        self.model = nn.Sequential(
            *[DecoderBlock(n_embd, n_head, mlp_bias, attn_proj_bias, attn_kv_bias, attn_dropout, mlp_dropout) \
            for _ in range(n_layer)],
            # decoder: (context_len, batch_size, n_embd)
            nn.LayerNorm(n_embd),
            # logits: (context_len, batch_size, vocab_size)
            nn.Linear(n_embd, vocab_size, bias=mlp_bias)
        )

    def forward(self, inputs: Tensor):
        # inputs: (batch_size, context_len)

        batch_size, context_len = inputs.shape
        # token_embedding: (batch_size, context_len, n_embd)
        token_embedding = self.token_embeddings(inputs)
        # positions: (batch_size, context_len)
        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
        # position_embedding: (batch_size, context_len, n_embd)
        position_embedding = self.position_embeddings(positions)
        # embedding: (batch_size, context_len, n_embd)
        embedding = token_embedding + position_embedding
        # embedding: (context_len, batch_size, n_embd)
        embedding = rearrange(embedding, 'b s d -> s b d')
        #embedding = embedding.permute(1, 0, 2)
        # output: (context_len, batch_size, vocab_size)
        return self.model(embedding)

    def get_num_params(self, non_embedding=True)->int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.get_params(non_embedding))
        return n_params

    def get_params(self, non_embedding=True):
        for p in self.parameters():
            if not non_embedding or \
                (p is not self.token_embeddings.weight and p is not self.position_embeddings.weight):
                yield p

    def weight_norm(self, non_embedding=True)->float:
        return torch.linalg.norm(torch.cat([p.view(-1) for p in self.get_params(non_embedding)])).item()

def get_model(n_layer: int, n_embd: int, n_head: int,
              vocab_size: int, context_length: int,
              mlp_bias: bool,
              attn_proj_bias: bool, # for projection layers in attention
              attn_kv_bias: bool, # for kv in attention
              attn_dropout: float, # dropout for attention layer
              mlp_dropout: float, # dropout for feedforward layer
              layer_norm_bias:bool,
              resid_dropout: float, # dropout for residual connection
              embed_dropout: float, # dropout for embedding layer
              ):
    return TinyTransformer(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
              vocab_size=vocab_size, context_len=context_length,
              mlp_bias=mlp_bias, attn_proj_bias=attn_proj_bias,
              attn_kv_bias=attn_kv_bias, attn_dropout=attn_dropout,
              mlp_dropout=mlp_dropout)