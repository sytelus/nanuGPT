# llm_sizing.py
from dataclasses import dataclass
from math import sqrt, ceil
from typing import Literal, Optional, Dict, Any, Tuple

Bytes = int

@dataclass
class LLMDesignInputs:
    target_params: int                # desired total parameters (trainable), including embeddings
    vocab_size: int                   # V
    context_length: int               # T
    linear_type: Literal["gelu", "swiglu"] = "swiglu"
    pos_encoding: Literal["rope", "learned"] = "rope"
    head_dim: Optional[int] = None    # default decided below (128 for large, else 64)
    ffn_mult: Optional[float] = None  # None -> choose default per linear_type
    weight_tying: bool = True         # tie lm_head to token embedding
    prefer_ratio: float = 128.0       # desired D/L ratio
    min_layers: int = 12
    max_layers: int = 160
    round_heads_to: int = 2           # enforce even #heads for kernel perf

@dataclass
class LLMDesignOutputs:
    head_dim: int
    n_heads: int
    n_layers: int
    hidden_size: int                  # D
    ffn_dim: int                      # F
    params_total: int                 # actual trainable params (including embeddings)
    params_embeddings: int            # token + (optional) position embeddings
    flops_forward: float              # attention+MLP (+ optional vocab) for B=1, full sequence
    flops_backward: float             # training backward-only FLOPs for B=1, full sequence
    memory_forward_bytes: Bytes       # mixed-precision inference-ish: 6B/param + KV cache
    memory_backward_bytes: Bytes      # AdamW training with checkpointing: ~18B/param + activations
    details: Dict[str, Any]           # extra internals

def _human(n: float, unit="B") -> str:
    s = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1024 and i < len(s) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {s[i]}{unit}"

def solve_hidden_given_layers(target_params: int, V: int, T_ctx: int, L: int,
                              mlp_coeff: float, pos_encoding: str) -> float:
    """
    Solve quadratic: A*L*D^2 + F_lin*D = target, ignoring tiny LN terms.
      A = (attention_per_layer + mlp_per_layer) / D^2 = (4 + mlp_coeff)
      F_lin = V + (T_ctx if learned positions else 0)
    Return positive root for D.
    """
    A = (4.0 + mlp_coeff)
    F_lin = V + (T_ctx if pos_encoding == "learned" else 0)
    # Quadratic: (A*L)*D^2 + F_lin*D - target_params = 0
    a = A * L
    b = F_lin
    c = -float(target_params)
    disc = b*b - 4*a*c
    if disc <= 0:
        return max(1.0, -b / (2*a))  # degenerate, fallback
    D = (-b + sqrt(disc)) / (2*a)
    return D

def round_hidden_heads(D_cont: float, head_dim: int, prefer_ratio: float, L: int,
                       round_heads_to: int) -> Tuple[int, int, int]:
    """
    Round hidden size to a multiple of head_dim and get heads.
    Optionally adjust L slightly to keep D/L close to prefer_ratio.
    """
    # Initial rounding of D
    D_rounded = max(head_dim, int(round(D_cont / head_dim)) * head_dim)
    n_heads = max(round_heads_to, int(round(D_rounded / head_dim)))
    # enforce even multiple (or multiple of round_heads_to)
    n_heads = int(ceil(n_heads / round_heads_to) * round_heads_to)
    D_rounded = n_heads * head_dim

    # Optionally nudge layers to keep D/L ~ prefer_ratio
    ideal_L = max(1, int(round(D_rounded / prefer_ratio)))
    # Do not drift too far from input L; allow small correction
    if abs(ideal_L - L) <= 2:
        L = ideal_L

    return D_rounded, n_heads, L

def count_params(L: int, D: int, F: int, V: int, T_ctx: int,
                 pos_encoding: str, linear_type: str) -> Tuple[int, int, Dict[str, int]]:
    """
    Returns (total_params, embedding_params, breakdown)
    """
    # Embedding parameters (token + maybe positional)
    pos_params = D * T_ctx if pos_encoding == "learned" else 0
    emb_params = V * D + pos_params

    # Per-layer params
    attn_params_per = 4 * D * D                    # Q/K/V/O
    if linear_type == "swiglu":
        mlp_params_per = 3 * D * F                 # gated: 2 up/gate + 1 down
    else:
        mlp_params_per = 2 * D * F                 # GELU/ReLU
    norm_params_per = 2 * D                        # two norms per block
    per_layer = attn_params_per + mlp_params_per + norm_params_per

    total = emb_params + L * per_layer
    return total, emb_params, {
        "attn_per_layer": attn_params_per,
        "mlp_per_layer": mlp_params_per,
        "norm_per_layer": norm_params_per
    }

def estimate_flops(B: int, T: int, L: int, D: int, F: int, n_heads: int, head_dim: int,
                   linear_type: str, include_vocab: bool, V: int) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (forward_flops, backward_flops, breakdown) for full sequence (prefill) with batch B.
    Formulas per JAX scaling notes.
    """
    # Attention train FLOPs per layer (MHA): 24*B*T*D^2 + 12*B*T^2*D
    attn_train_per = 24.0 * B * T * (D * D) + 12.0 * B * (T**2) * D
    # Forward ≈ 1/3, Backward ≈ 2/3
    attn_fwd_per = attn_train_per / 3.0
    attn_bwd_per = attn_train_per - attn_fwd_per

    # MLP train FLOPs per layer
    if linear_type == "swiglu":
        mlp_train_per = 18.0 * B * T * D * F
    else:
        mlp_train_per = 12.0 * B * T * D * F
    mlp_fwd_per = mlp_train_per / 3.0
    mlp_bwd_per = mlp_train_per - mlp_fwd_per

    # Sum across L layers
    fwd = L * (attn_fwd_per + mlp_fwd_per)
    bwd = L * (attn_bwd_per + mlp_bwd_per)

    # Optional vocab projection (training forward is ~4 B T D V in that source)
    vocab_fwd = 4.0 * B * T * D * V if include_vocab else 0.0
    vocab_bwd = 8.0 * B * T * D * V if include_vocab else 0.0  # training-only
    fwd += vocab_fwd
    bwd += vocab_bwd

    return fwd, bwd, {
        "attn_fwd": L * attn_fwd_per,
        "mlp_fwd": L * mlp_fwd_per,
        "attn_bwd": L * attn_bwd_per,
        "mlp_bwd": L * mlp_bwd_per,
        "vocab_fwd": vocab_fwd,
        "vocab_bwd": vocab_bwd
    }

def estimate_memory_forward_inference(params_total: int, L: int, T: int,
                                      n_kv_heads: int, head_dim: int,
                                      dtype_bytes: int = 2, keep_master_fp32: bool = True) -> int:
    """
    Mixed-precision inference-ish:
      Weights: 6 bytes/param (HF rule-of-thumb) if keep_master_fp32 else ~2 bytes/param.
      KV cache: 2 (K and V) * L * T * n_kv_heads * head_dim * dtype_bytes.
    """
    weight_bytes = (6 if keep_master_fp32 else 2) * params_total
    kv_cache_bytes = 2 * L * T * n_kv_heads * head_dim * dtype_bytes
    return weight_bytes + kv_cache_bytes

def estimate_memory_backward_training(params_total: int, B: int, T: int, L: int, D: int,
                                      dtype_bytes: int = 2) -> int:
    """
    AdamW + bf16 mixed precision with activation checkpointing (block remat).
      ~18 bytes per parameter (HF) + saved activations ≈ B * T * D * dtype_bytes * L
    """
    optimizer_bytes = 18 * params_total
    activations_bytes = B * T * D * dtype_bytes * L
    return optimizer_bytes + activations_bytes

def recommend_llm_config(cfg: LLMDesignInputs,
                         B_for_flops: int = 1,
                         include_vocab_in_flops: bool = False) -> LLMDesignOutputs:
    # Defaults for head_dim and ffn_mult
    if cfg.head_dim is None:
        cfg.head_dim = 128 if cfg.target_params >= 300_000_000 else 64
    if cfg.ffn_mult is None:
        cfg.ffn_mult = 3.5 if cfg.linear_type == "swiglu" else 4.0

    # MLP param coefficient per D^2 (used for solving hidden size)
    mlp_coeff = (3.0 * cfg.ffn_mult) if cfg.linear_type == "swiglu" else (2.0 * cfg.ffn_mult)

    best = None
    for L in range(cfg.min_layers, cfg.max_layers + 1):
        # Solve for D ignoring small terms first
        D_cont = solve_hidden_given_layers(cfg.target_params, cfg.vocab_size, cfg.context_length,
                                           L, mlp_coeff, cfg.pos_encoding)
        # Round D and possibly nudge L to keep D/L ~ prefer_ratio
        D_rounded, n_heads, L_adj = round_hidden_heads(D_cont, cfg.head_dim, cfg.prefer_ratio, L, cfg.round_heads_to)
        F = int(round(cfg.ffn_mult * D_rounded))
        # Count actual params
        params_total, params_emb, breakdown = count_params(L_adj, D_rounded, F, cfg.vocab_size,
                                                           cfg.context_length, cfg.pos_encoding, cfg.linear_type)
        # Keep the configuration that best matches the target and ratio
        err = abs(params_total - cfg.target_params)
        ratio_err = abs((D_rounded / max(1, L_adj)) - cfg.prefer_ratio)
        score = (err, ratio_err)
        candidate = (score, (L_adj, D_rounded, n_heads, F, params_total, params_emb, breakdown))
        if best is None or candidate[0] < best[0]:
            best = candidate

    # Unpack best choice
    (_, (L, D, n_heads, F, params_total, params_emb, breakdown)) = best

    # FLOPs (B=1 by default) for full sequence length
    fwd_flops, bwd_flops, flops_breakdown = estimate_flops(
        B=B_for_flops, T=cfg.context_length, L=L, D=D, F=F,
        n_heads=n_heads, head_dim=cfg.head_dim,
        linear_type=cfg.linear_type, include_vocab=include_vocab_in_flops,
        V=cfg.vocab_size
    )

    # Memory estimates
    mem_fwd = estimate_memory_forward_inference(params_total, L, cfg.context_length,
                                                n_kv_heads=n_heads, head_dim=cfg.head_dim,
                                                dtype_bytes=2, keep_master_fp32=True)
    mem_bwd = estimate_memory_backward_training(params_total, B=1, T=cfg.context_length, L=L, D=D,
                                                dtype_bytes=2)

    return LLMDesignOutputs(
        head_dim=cfg.head_dim,
        n_heads=n_heads,
        n_layers=L,
        hidden_size=D,
        ffn_dim=F,
        params_total=params_total,
        params_embeddings=params_emb,
        flops_forward=fwd_flops,
        flops_backward=bwd_flops,
        memory_forward_bytes=mem_fwd,
        memory_backward_bytes=mem_bwd,
        details={
            "attn_params_per_layer": breakdown["attn_per_layer"],
            "mlp_params_per_layer": breakdown["mlp_per_layer"],
            "norm_params_per_layer": breakdown["norm_per_layer"],
            "prefer_ratio": cfg.prefer_ratio,
            "ratio_D_over_L": D / max(1, L),
            "includes_vocab_in_flops": include_vocab_in_flops,
            "notes": {
                "formulas_source": "JAX How To Scale Your Model — Transformer Math",
                "memory_source": "HF Transformers perf train on single GPU",
                "assumptions": "MHA (N=K), weight tying, no biases, bf16 activations"
            }
        }
    )

def pretty_print(design: LLMDesignOutputs) -> None:
    print("=== Recommended Architecture ===")
    print(f"Layers (L):         {design.n_layers}")
    print(f"Hidden size (D):    {design.hidden_size}")
    print(f"Head dim (H):       {design.head_dim}")
    print(f"Heads (N):          {design.n_heads}")
    print(f"FFN dim (F):        {design.ffn_dim}")
    print()
    print("=== Parameters ===")
    print(f"Total params:       {design.params_total:,}")
    print(f"Embedding params:   {design.params_embeddings:,}")
    print()
    print("=== FLOPs (B=1, full sequence) ===")
    print(f"Forward FLOPs:      {design.flops_forward:,.0f}")
    print(f"Backward FLOPs:     {design.flops_backward:,.0f}")
    print()
    print("=== Memory Estimates ===")
    print(f"Forward (inference-ish): {_human(design.memory_forward_bytes)} "
          f"({design.memory_forward_bytes:,} bytes)")
    print(f"Backward (training):     {_human(design.memory_backward_bytes)} "
          f"({design.memory_backward_bytes:,} bytes)")

if __name__ == "__main__":
    cfg = LLMDesignInputs(
        target_params=7_000_000_000,  # 7B
        vocab_size=32_000,
        context_length=8192,
        linear_type="swiglu",
        pos_encoding="rope",
    )
    design = recommend_llm_config(cfg, B_for_flops=1, include_vocab_in_flops=False)
    pretty_print(design)
