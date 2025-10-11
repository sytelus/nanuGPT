"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from typing import Optional, Tuple, Callable

import math
import inspect
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanugpt import common

class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int =  50304 # GPT-2 vocab_size of 50257, this can be padded up to nearest multiple of 64 which is 50304 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_bias: bool = True # default
    attn_proj_bias: bool = True # default
    attn_kv_bias: bool = False  # Karpathy has this to PyTorch default which is True
    layer_norm_bias: bool = True
    attn_dropout: float = 0.0 # applied on softmax of QK^T
    mlp_dropout: float = 0.0
    resid_dropout: float = 0.0
    embed_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_lm_head: bool = True  # tie token embedding and lm_head weights by default

# GPT-2 124M-like config
CONFIG_124M = GPTConfig(
    n_embd=768,
    n_layer=12,
    n_head=12,
    tie_lm_head=False, # for comparing with llama
)

# GPT-2 355M-like config
CONFIG_345M = GPTConfig(
    n_embd=1024,
    n_layer=24,
    n_head=16,
    tie_lm_head=False, # for comparing with llama
)

# GPT-3 1.3B-like config
CONFIG_1p3B = GPTConfig(
    n_embd=2048,
    n_layer=24,
    n_head=16,
    block_size=2048,
    tie_lm_head=False, # for comparing with llama
)

# llama 8B-like config
CONFIG_8B = GPTConfig(
    n_embd=4096,
    n_layer=32,
    n_head=32,
    block_size=4096,
    tie_lm_head=False, # for comparing with llama
)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias, eps=1e-05):
        # batch normalization computes avg across batch of data for normalizing but
        # this doesn't work if batch is too small. LayerNorm normalizes across the
        # feature dimension instead (n_embd) and is independent of batch size.
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)

class CausalSelfAttention(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()

        # ensure each head gets same bits of the embedding dim
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_kv_bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_proj_bias)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1 # type: ignore

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash_attn_dropout_val = config.attn_dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logging.warning("Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # TODO: change name from bias to tri, this is a causal mask and "bias" is confusing name here
            # register_buffer ensures that this is not a parameter even though it is a tensor
            # unfortunately, this will get saved in the model state dict, it might be good to handle this
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, *,
                attention_mask: Optional[torch.Tensor] = None,):
        batch_size, seq_len, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # self.c_attn(x): (batch_size, seq_len, n_embd) -> (batch_size, seq_len, 3 * n_embd)
        # .split(self.n_embd, dim=2): (batch_size, seq_len, 3 * n_embd) -> (batch_size, seq_len, n_embd), (batch_size, seq_len, n_embd), (batch_size, seq_len, n_embd)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_heads, seq_len, head_size)
        q = q.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_heads, seq_len, head_size)
        v = v.view(batch_size, seq_len, self.n_head, n_embd // self.n_head).transpose(1, 2) # (batch_size, n_heads, seq_len, head_size)

        # causal self-attention; Self-attend: (batch_size, n_heads, seq_len, head_size) x (batch_size, n_heads, head_size, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                    attn_mask=None,
                    dropout_p=self.flash_attn_dropout_val if self.training else 0.0,
                    is_causal=True) # TODO: is_causal shouldn't be true for inference!
        else:
            # manual implementation of attention

            # q, k, v: (batch_size, n_heads, seq_len, head_size)
            # q @ k.transpose(-2, -1): (batch_size, n_heads, seq_len, head_size) x (batch_size, n_heads, head_size, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
            # product of q and k produces matrix where each element (i, j) contains the dot product of sequence i with sequence j.
            # This dot product is across all embeddings of those two sequences.
            # @ is matrix multiplication
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (batch_size, n_heads, seq_len, seq_len)
            # we use fixed triangular matrix and where it is 0, we set the attention to -inf so that softmax will ignore those values
            att = att.masked_fill(self.bias[:,:,:seq_len,:seq_len] == 0, float('-inf'))

            # softmax on last axis (seq_len) so that values in each row sum to 1 in attention matrix.
            # The way this works is, we sum up exponents of values in each row and divide each value by that sum.
            # So, the output of softmax has same shape as input, and each row sums to 1 because we set dim=-1 which means
            # each value exponent in column gets divided by sum of exponents of all values in that column.
            # At this element in each row in attn contains float indicating importance of that column for that row.
            att = F.softmax(att, dim=-1) # (batch_size, n_heads, seq_len, seq_len)
            # attn dropout will most likely turn off noisy values and therefore strenghten the signal
            att = self.attn_dropout(att)

            # now that each row sums upto 1, we can multiply row with column to get weighted sum of values in each column
            # The way tensor multiplication works is that you just pretend there are only last two dimentions and
            # produce same shape as normal multiplication to replace last two dimentions. Other way to think is that
            # each batch and each head having its own matrix which gets multiplied independently.
            # Each row in v is for one token so there same number of rows in v as there are tokens in the input
            # Each column in v is for one head so there same number of columns in v as there are heads
            # When we multiply the attention matrix with v, we are doing weighted sum of the values in v for each head.
            # The weight for each token is sequence is the attention value for that token in that sequence.
            # The output is one row for each token in the input sequence and one column for each head.
            y = att @ v # (batch_size, n_heads, seq_len, seq_len) x (batch_size, n_heads, seq_len, head_size) -> (batch_size, n_heads, seq_len, head_size)

        # put back the head dimension together at the last and join them to produce n_embd
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd) # (batch_size, seq_len, d_model)

        # output projection basically does the "ensemble" of all heads into one head
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.mlp_bias)
        self.gelu    = NewGELU()    # nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.mlp_dropout) if config.mlp_dropout else None
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1 # type: ignore

    def forward(self, x):
        # notice that each token in the sequence is processed independently
        # basically embedding for each token gets projected by FFN weight matrix to produce 4X bigger embedding
        # this is why FFN can be trivially parallelized across tokens in the sequence
        x = self.c_fc(x) # (batch_size, seq_len, n_embd) -> (batch_size, seq_len, 4 * n_embd)
        x = self.gelu(x) # (batch_size, seq_len, 4 * n_embd) -> (batch_size, seq_len, 4 * n_embd)
        x = self.c_proj(x) # (batch_size, seq_len, 4 * n_embd) -> (batch_size, seq_len, n_embd)
        if self.dropout:
            x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        # Original transformer had two layer norms, one after self-attention and one after MLP.
        # It also had two residual connections, one after self-attention and one after MLP.
        # GPT2 has two layer norms per block, one at the start and other before MLP.
        # So the layer norms in GPT are moved to start of sub-blocks instead of end.
        # GPT2 also has two residual connections, one after self-attention and other after MLP.
        # Residual connections applied before layer norms are applied so input travels intact.
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.layer_norm_bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.layer_norm_bias)
        self.mlp = MLP(config)

    def forward(self, x,
                attention_mask: Optional[torch.Tensor] = None,
                ):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config:GPTConfig, get_loss: Optional[common.GetLossType]):
        super().__init__()
        self.config = config
        self.get_loss = get_loss

        modules:dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # n_embd === hidden_size === d_model
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.layer_norm_bias),
        )
        if config.embed_dropout:
            modules['embed_dropout'] = nn.Dropout(config.embed_dropout)
        self.transformer = nn.ModuleDict(modules)

        # LM head is last layer which projects the final hidden state to vocab_size
        # we keep bias False to match with embedding layer which has no bias
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_lm_head:
            self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights # type: ignore

            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            # https://paperswithcode.com/method/weight-tying
            self.transformer.wte.weight = self.lm_head.weight # type: ignore

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

        # below was removed in llm.c update
        # apply special scaled init to the residual projections, per GPT-2 paper
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=self.config.initializer_range/math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        # overall it seems that Karpathy's init choses smaller weights than PyTorch default

        # default init for linear layer in Pytorch is kaiming_uniform_ which is 1/sqrt(fan_in)
        # for fan_in=768, default stddev would be 0.036.
        # TODO: not sure if below is good idea and is independent of the size of the model
        if isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = self.config.initializer_range if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else self.config.initializer_range/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # default init for embedding layer in Pytorch is init.normal_ which is N(0, 1)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range, generator=self.init_rng)

    def forward(self,
                idx:torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_logits: bool = True,
                only_last=False,
                attention_mask: Optional[torch.Tensor] = None,
                is_first_microbatch: Optional[bool]=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:  # logits, loss, num_correct

        device = idx.device
        batch_size, seq_len = idx.size()
        assert seq_len <= self.config.block_size, f"Cannot forward sequence of length {seq_len} but context_len is only {self.config.block_size}"

        # create pos vector of same size as input seq
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device) # shape (seq_len)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # type: ignore # token embeddings of shape (batch_size, seq_len, n_embd)
        pos_emb = self.transformer.wpe(pos) # type: ignore # position embeddings of shape (seq_len, n_embd)
        if self.config.embed_dropout:
            x = self.transformer.embed_dropout(tok_emb + pos_emb) # type: ignore
        else:
            x = tok_emb + pos_emb
        for block in self.transformer.h: # type: ignore
            x = block(x, attention_mask=attention_mask) # (batch_size, seq_len, n_embd)

        # apply layer norm. Typically layer norm is applied before and after attention but not after MLP.
        x = self.transformer.ln_f(x) # type:ignore # [batch, seq_len, emb_dim]

        if not only_last:
            logits:torch.Tensor = self.lm_head(x) # [batch, seq_len, vocab_size]
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
             # note: using list [-1] to preserve the time dim
            logits:torch.Tensor = self.lm_head(x[:, [-1], :]) # [batch, 1, vocab_size]

        loss:Optional[torch.Tensor] = None
        if labels is not None:
            assert self.get_loss is not None, "Loss function is not defined"
            loss, correct = self.get_loss(logits, labels)
            # keeping logits around may unnecessarily consume a lot of memory  (atleast 1GB for 124M params)
            return (logits, loss, correct) if return_logits else (None, loss, correct)

        return logits, None, None


def get_model(
                vocab_size: int,
                get_loss: Optional[common.GetLossType],

                n_layer: int, n_embd: int, n_head: int,
                context_length: int,

                mlp_bias=False,
                attn_proj_bias=False, # for projection layers in attention
                attn_kv_bias=False, # for kv in attention
                layer_norm_bias=False, # for layer norm

                resid_dropout=0.0, # dropout for residual in attention
                embed_dropout=0.0, # dropout for embedding layer
                attn_dropout=0.0, # dropout for attention layer
                mlp_dropout=0.0, # dropout for feedforward layer

                initializer_range=0.02, # defaults to 0.02, The standard deviation of the truncated_normal_initializer for initializing all weight matrices
                tie_lm_head=True,
              ):

    gpt_config = GPTConfig(
                            block_size=context_length,
                            vocab_size=vocab_size,
                            n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            mlp_bias=mlp_bias,
                            attn_proj_bias=attn_proj_bias,
                            attn_kv_bias=attn_kv_bias,
                            layer_norm_bias=layer_norm_bias,
                            initializer_range=initializer_range,
                            attn_dropout=attn_dropout,
                            mlp_dropout=mlp_dropout,
                            resid_dropout=resid_dropout,
                            embed_dropout=embed_dropout,
                            tie_lm_head=tie_lm_head,
                            )

    return GPT(gpt_config, get_loss)
