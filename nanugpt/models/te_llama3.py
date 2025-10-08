from typing import Optional
import math
from dataclasses import dataclass
import torch
from torch import nn


import torch.nn.init as init
from functools import partial

from transformer_engine import pytorch as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

@dataclass
class LlamaConfig:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    swiglu_multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000 # TODO: original is 500000 but because of https://github.com/NVIDIA/TransformerEngine/issues/1245 we set lower
    max_seq_len: int = 4096
    # defined later by tokenizer
    vocab_size: int = -1

TE_LLAMA_TINY_CONFIG = LlamaConfig(
    d_model=1024,
    n_layers=4,
    n_heads=8,
    n_kv_heads=None,
    vocab_size=-1,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_360M_CONFIG = LlamaConfig(
    d_model=1024,
    n_layers=24,
    n_heads=8,
    n_kv_heads=None,
    vocab_size=-1,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_2B_CONFIG = LlamaConfig(
    d_model=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=None,
    vocab_size=-1,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_3B_CONFIG = LlamaConfig(
    d_model=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=None,
    vocab_size=-1,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1,
)

TE_LLAMA_8B_CONFIG = LlamaConfig(
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=-1,
    swiglu_multiple_of=1024,
    ffn_dim_multiplier=1.3,
)

class TeLlama3DecoderLayer(te.TransformerLayer):
    def __init__(self, d_model:int, ffn_dim:int, layer_num:int,
                 n_heads:int, n_kv_heads:int, max_seq_len:int,
                 rms_norm_eps:float, rope_theta:float):
        super().__init__(
            hidden_size=d_model,
            ffn_hidden_size=ffn_dim,
            num_attention_heads=n_heads,
            bias=False,
            layernorm_epsilon=rms_norm_eps,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            fuse_qkv_params=True, # set to False in NVidia tutorial, # TODO: check if needed
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=n_kv_heads,
            # init_method=torch_init_method, # TODO: check if needed
            # output_layer_init_method=torch_init_method, # TODO: check if needed
            layer_number=layer_num,
            self_attn_mask_type="causal", # default
            seq_length=math.ceil(max_seq_len // 128) * 128,
            #params_dtype=torch.bfloat16,

        )
        te_rope = RotaryPositionEmbedding(d_model//n_heads, rotary_base=rope_theta)
        self.te_rope_emb = te_rope(max_seq_len=max_seq_len).cuda()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask,):
        return super().forward(hidden_states, attention_mask=attention_mask, rotary_pos_emb=self.te_rope_emb)


class TeLlama3Model(nn.Module):
    def __init__(self, d_model:int, layer_num:int,
                 n_heads:int, n_kv_heads:int, max_seq_len:int,
                 rms_norm_eps:float, n_layers:int, vocab_size:int,
                 ffn_dim_multiplier:Optional[float],
                 swiglu_multiple_of:int,
                 rope_theta:float,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        ffn_dim = int(8 * d_model / 3)
        if ffn_dim_multiplier is not None:
            ffn_dim = int(ffn_dim_multiplier * ffn_dim)
        ffn_dim = swiglu_multiple_of * (
            (ffn_dim + swiglu_multiple_of - 1) // swiglu_multiple_of
        )

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList()
        torch_init_method = partial(init.kaiming_uniform_, a=5)

        for layer_num in range(n_layers):
            transformer_block = TeLlama3DecoderLayer(
                d_model=d_model,
                ffn_dim=ffn_dim,
                layer_num=layer_num,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
            )
            self.layers.append(transformer_block)

        self.lm_head = te.LayerNormLinear(d_model, vocab_size, bias=False,
                                 normalization='RMSNorm', eps=rms_norm_eps)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask=None,):
        h = self.tok_embeddings(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
        logits = self.lm_head(h)
        return logits
