from einops import rearrange, repeat
import torch
from torch import nn, Tensor

class DecoderBlock(torch.nn.Module):
  def __init__(self, dim_model: int, n_heads: int):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
    self.self_attn_norm = nn.LayerNorm(dim_model)
    self.ffn = nn.Sequential(
        nn.Linear(dim_model, dim_model * 4),
        nn.GELU(),
        nn.Linear(dim_model * 4, dim_model)
    )
    self.ffn_norm = nn.LayerNorm(dim_model)

  def forward(self, x: Tensor):
    # x: (context_len, batch_size, dim_model)
    # attn_mask: (context_len, context_len)
    # attn_mask = torch.full(
    #     (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    # )
    # attn_mask = torch.triu(attn_mask, diagonal=1)

    attn_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[0], device=x.device)

    # a1: (context_len, batch_size, dim_model)
    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    a1 = self.self_attn_norm (x + a1)
    # a1: (context_len, batch_size, dim_model)
    a2 = self.ffn(a1)
    a2 = self.ffn_norm(a1 + a2)

    return a2

class Transformer(torch.nn.Module):
  def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int):
    super().__init__()

    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    self.model = nn.Sequential(
        *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
        # decoder: (context_len, batch_size, dim_model)
        nn.LayerNorm(dim_model),
        # logits: (context_len, batch_size, num_tokens)
        nn.Linear(dim_model, num_tokens)
    )

  def forward(self, inputs: Tensor):
    # inputs: (batch_size, context_len)

    batch_size, context_len = inputs.shape
    # token_embedding: (batch_size, context_len, dim_model)
    token_embedding = self.token_embeddings(inputs)
    # positions: (batch_size, context_len)
    positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
    # position_embedding: (batch_size, context_len, dim_model)
    position_embedding = self.position_embeddings(positions)
    # embedding: (batch_size, context_len, dim_model)
    embedding = token_embedding + position_embedding
    # embedding: (context_len, batch_size, dim_model)
    embedding = rearrange(embedding, 'b s d -> s b d')
    #embedding = embedding.permute(1, 0, 2)
    # output: (context_len, batch_size, num_tokens)
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
              ffn_bias: bool,
              attn_proj_bias: bool, # for projection layers in attention
              attn_kv_bias: bool, # for kv in attention
              attn_dropout: float, # dropout for attention layer
              ffn_dropout: float # dropout for feedforward layer
              ):
  return Transformer(n_layer, n_embd, n_head,
              vocab_size, context_length)