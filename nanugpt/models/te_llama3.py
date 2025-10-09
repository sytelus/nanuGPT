# Implements Llama 2/3 architecture using NVIDIA Transformer Engine
# References:
# Config: https://huggingface.co/unsloth/llama-3-8b/blob/main/config.json
# Karpathy llm.c: https://github.com/karpathy/llm.c/blob/master/train_llama3.py
# TE reference: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html
# B200 benchmarking: https://github.com/HoomanRamezani/b200_benchmarking/blob/main/bench_te.py

from typing import Optional, Tuple, Callable
import math
from dataclasses import dataclass
import torch
from torch import nn

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformer_engine import pytorch as te
from transformer_engine.pytorch.attention import RotaryPositionEmbedding

from nanugpt import common

@dataclass
class LlamaConfig:
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_heads: int = 8
    swiglu_multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000 # TODO: original is 500000 but because of https://github.com/NVIDIA/TransformerEngine/issues/1245 we set lower
    block_size: int = 4096
    # defined later by tokenizer
    vocab_size: int = 32000 # 128256 for Llama3
    # ffn_dim is 14336 for llama3-8b, 11008 for llama-2-7b
    initializer_range: float = 0.023 # 0.02 in HF, 0.023 in TE

TE_LLAMA_TINY_CONFIG = LlamaConfig(
    n_embd=1024,
    n_layer=4,
    n_head=8,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_360M_CONFIG = LlamaConfig(
    n_embd=1024,
    n_layer=24,
    n_head=8,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_2B_CONFIG = LlamaConfig(
    n_embd=2048,
    n_layer=24,
    n_head=16,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1.3,
)

TE_LLAMA_3B_CONFIG = LlamaConfig(
    n_embd=3072,
    n_layer=28,
    n_head=24,
    swiglu_multiple_of=256,
    ffn_dim_multiplier=1,
)

TE_LLAMA_8B_CONFIG = LlamaConfig(
    n_embd=4096,
    n_layer=32,
    n_head=32,
    swiglu_multiple_of=1024,
    ffn_dim_multiplier=1.3,
)

class TeLlama3DecoderLayer(te.TransformerLayer):
    def __init__(self, n_embd:int, ffn_dim:int, layer_num:int,
                 n_head:int, n_kv_heads:int, block_size:int,
                 rms_norm_eps:float, rope_theta:float):
        super().__init__(
            hidden_size=n_embd,
            ffn_hidden_size=ffn_dim,
            num_attention_heads=n_head,
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
            seq_length=math.ceil(block_size // 128) * 128,
            #params_dtype=torch.bfloat16,

        )
        te_rope = RotaryPositionEmbedding(n_embd//n_head, rotary_base=rope_theta)
        self.te_rope_emb = te_rope(max_seq_len=block_size).cuda()

    def forward(self, hidden_states: torch.Tensor, attention_mask:Optional[torch.Tensor],
                is_first_microbatch: Optional[bool]):
        return super().forward(hidden_states, attention_mask=attention_mask,
                               rotary_pos_emb=self.te_rope_emb,
                               is_first_microbatch=is_first_microbatch)




class TeLlama3Model(nn.Module):
    def __init__(self, config: LlamaConfig, get_loss: Optional[common.GetLossType]):
        super().__init__()
        self.config = config
        self.get_loss = get_loss

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = torch.nn.ModuleList()

        # Llama3 uses 3.5 x n_embd for hidden layer size of SwiGLU
        # 3.5 = 8/3*1.3 (1.3 is ffn_dim_multiplier)
        # this then gets rounded to multiple of swiglu_multiple_of (usually 256 or 1024)
        # llama2 used 8/3 x n_embd so we can set ffn_dim_multiplier to 1.0 to get that
        ffn_dim = int(8 * config.n_embd / 3)
        if config.ffn_dim_multiplier is not None:
            ffn_dim = int(config.ffn_dim_multiplier * ffn_dim)
        ffn_dim = config.swiglu_multiple_of * (
            (ffn_dim + config.swiglu_multiple_of - 1) // config.swiglu_multiple_of
        )

        for layer_num in range(config.n_layer):
            transformer_block = TeLlama3DecoderLayer(
                n_embd=config.n_embd,
                ffn_dim=ffn_dim,
                layer_num=layer_num,
                n_head=config.n_head,
                n_kv_heads=config.n_kv_heads,
                block_size=config.block_size,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
            )
            self.layers.append(transformer_block)

        self.lm_head = te.LayerNormLinear(config.n_embd, config.vocab_size, bias=False,
                                 normalization='RMSNorm', eps=config.rms_norm_eps)

        # use rng for weight initialization
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Pytorch default for linear/swiglue is kaiming_uniform_ which is 1/sqrt(fan_in)
        # but HF and TE use init.normal_ with stddev = initializer_range
        # HF default initializer_range is 0.02 which is close to 1/sqrt(2500)
        # TE default initializer_range is 0.023
        # for fan_in=768, default stddev would be 0.036
        # default init for embedding layer in Pytorch is init.normal_ which is N(0, 1)

        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range, generator=self.init_rng)
        # leave other initializations to TE defaults

    def forward(self,
                idx: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_logits: bool = True,
                only_last=False,
                attention_mask: Optional[torch.Tensor] = None,
                is_first_microbatch: Optional[bool]=None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:  # logits, loss, num_correct
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, is_first_microbatch=is_first_microbatch)

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
                n_kv_heads:int,
                ffn_dim_multiplier: Optional[float],
                swiglu_multiple_of: int,
                rope_theta: float,
                rms_norm_eps: float,
                initializer_range: float=0.023,
              ):

    config = LlamaConfig(
                            block_size=context_length,
                            vocab_size=vocab_size,
                            n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            n_kv_heads=n_kv_heads,
                            ffn_dim_multiplier=ffn_dim_multiplier,
                            swiglu_multiple_of=swiglu_multiple_of,
                            rope_theta=rope_theta,
                            rms_norm_eps=rms_norm_eps,
                            initializer_range=initializer_range,
                        )

    return TeLlama3Model(config, get_loss)