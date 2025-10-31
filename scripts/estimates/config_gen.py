# llm_sizing.py  (updated)
from dataclasses import dataclass
from math import sqrt, ceil, gcd
from typing import Literal, Optional, Dict, Any, Tuple

import torch

def _lcm(a: int, b: int) -> int:
    return abs(a*b) // gcd(a, b) if a and b else 0

def _nearest_multiple(x: int, m: int) -> int:
    if m <= 0: return x
    return max(m, int(round(x / m)) * m)

Bytes = int

@dataclass
class LLMDesignInputs:
    target_params: int                # desired total trainable params (incl. embeddings)
    vocab_size: int                   # V
    context_length: int               # T
    linear_type: Literal["gelu", "swiglu"] = "swiglu"
    pos_encoding: Literal["rope", "learned"] = "rope"
    head_dim: Optional[int] = None    # default decided below
    ffn_mult: Optional[float] = None  # None -> default per linear_type
    weight_tying: bool = True         # tie lm_head to token embedding
    # New: heads≈layers preference
    tie_heads_to_layers: bool = True
    # GQA / FA2-related options
    n_kv_heads: Optional[int] = None  # if set, require n_heads % n_kv_heads == 0
    rotary_dim: Optional[int] = None  # for RoPE; if None and rope: set to head_dim
    enforce_fa2: bool = True          # round head_dim to multiple-of-8 etc.
    paged_kv: bool = False
    page_block_size: int = 256        # must be multiple of 256 if paged_kv
    # Search bounds
    min_layers: int = 12
    max_layers: int = 160
    round_heads_to: int = 2           # enforce even heads for kernel perf

@dataclass
class LLMDesignOutputs:
    head_dim: int
    n_heads: int
    n_layers: int
    hidden_size: int                  # D
    ffn_dim: int                      # F
    params_total: int                 # actual trainable params (including embeddings)
    params_embeddings: int            # token + (optional) position embeddings
    flops_forward: float              # B=1, full sequence (prefill), no vocab projection by default
    flops_backward: float             # B=1, training backward-only
    memory_forward_bytes: Bytes       # mixed-precision inference-ish: 6B/param + KV cache
    memory_backward_bytes: Bytes      # AdamW training with checkpointing: ~18B/param + activations
    details: Dict[str, Any]

def _human(n: float, unit="B") -> str:
    s = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1024 and i < len(s) - 1:
        n /= 1024
        i += 1
    return f"{n:.2f} {s[i]}{unit}"

def _solve_hidden_given_layers(target_params: int, V: int, T_ctx: int, L: int,
                               mlp_coeff: float, pos_encoding: str) -> float:
    # Solve (A*L)*D^2 + F_lin*D - target_params = 0
    A = (4.0 + mlp_coeff)
    F_lin = V + (T_ctx if pos_encoding == "learned" else 0)
    a = A * L
    b = F_lin
    c = -float(target_params)
    disc = b*b - 4*a*c
    if disc <= 0:
        return max(1.0, -b / (2*a))
    return (-b + sqrt(disc)) / (2*a)

def _count_params(L: int, D: int, F: int, V: int, T_ctx: int,
                  pos_encoding: str, linear_type: str) -> Tuple[int, int, Dict[str, int]]:
    pos_params = D * T_ctx if pos_encoding == "learned" else 0
    emb_params = V * D + pos_params
    attn_params_per = 4 * D * D
    mlp_params_per = (3 * D * F) if linear_type == "swiglu" else (2 * D * F)
    norm_params_per = 2 * D
    per_layer = attn_params_per + mlp_params_per + norm_params_per
    total = emb_params + L * per_layer
    return total, emb_params, {
        "attn_per_layer": attn_params_per,
        "mlp_per_layer": mlp_params_per,
        "norm_per_layer": norm_params_per
    }

def _estimate_flops(B: int, T: int, L: int, D: int, F: int,
                    linear_type: str, include_vocab: bool, V: int) -> Tuple[float, float, Dict[str, float]]:
    # Attention train FLOPs per layer: 24*B*T*D^2 + 12*B*T^2*D
    attn_train_per = 24.0 * B * T * (D * D) + 12.0 * B * (T**2) * D
    attn_fwd_per = attn_train_per / 3.0
    attn_bwd_per = attn_train_per - attn_fwd_per
    # MLP
    mlp_train_per = (18.0 if linear_type == "swiglu" else 12.0) * B * T * D * F
    mlp_fwd_per = mlp_train_per / 3.0
    mlp_bwd_per = mlp_train_per - mlp_fwd_per
    fwd = L * (attn_fwd_per + mlp_fwd_per)
    bwd = L * (attn_bwd_per + mlp_bwd_per)
    # optional vocab projection
    if include_vocab:
        fwd += 4.0 * B * T * D * V
        bwd += 8.0 * B * T * D * V
    return fwd, bwd, {
        "attn_fwd": L * attn_fwd_per,
        "mlp_fwd": L * mlp_fwd_per,
        "attn_bwd": L * attn_bwd_per,
        "mlp_bwd": L * mlp_bwd_per,
    }

def _estimate_memory_forward_inference(params_total: int, L: int, T: int,
                                       n_kv_heads: int, head_dim: int,
                                       dtype_bytes: int = 2, keep_master_fp32: bool = True) -> int:
    weight_bytes = (6 if keep_master_fp32 else 2) * params_total
    kv_cache_bytes = 2 * L * T * n_kv_heads * head_dim * dtype_bytes  # K and V
    return weight_bytes + kv_cache_bytes

def _estimate_memory_backward_training(params_total: int, B: int, T: int, L: int, D: int,
                                       dtype_bytes: int = 2) -> int:
    optimizer_bytes = 18 * params_total   # AdamW + bf16 master/grads
    activations_bytes = B * T * D * dtype_bytes * L  # checkpointing: ~one tensor per block
    return optimizer_bytes + activations_bytes

def recommend_llm_config(
    cfg: LLMDesignInputs,
    B_for_flops: int = 1,
    include_vocab_in_flops: bool = False
) -> LLMDesignOutputs:

    # Defaults
    if cfg.head_dim is None:
        cfg.head_dim = 128 if cfg.target_params >= 300_000_000 else 64
    if cfg.ffn_mult is None:
        cfg.ffn_mult = 3.5 if cfg.linear_type == "swiglu" else 4.0

    # FA2-friendly rounding/checks
    fa2_notes = []
    if cfg.enforce_fa2:
        if cfg.head_dim % 8 != 0:
            new_hd = _nearest_multiple(cfg.head_dim, 8)
            fa2_notes.append(f"Rounded head_dim from {cfg.head_dim} to {new_hd} (multiple-of-8 for kernels).")
            cfg.head_dim = new_hd
        if cfg.head_dim > 256:
            fa2_notes.append(f"Clamped head_dim {cfg.head_dim} to 256 (FlashAttention-2 limit).")
            cfg.head_dim = 256
        if cfg.paged_kv and (cfg.page_block_size % 256 != 0):
            new_pb = _nearest_multiple(cfg.page_block_size, 256)
            fa2_notes.append(f"Rounded page_block_size to {new_pb} (paged KV requires multiple of 256).")
            cfg.page_block_size = new_pb

    # Rotary defaults
    if cfg.pos_encoding == "rope" and cfg.rotary_dim is None:
        cfg.rotary_dim = cfg.head_dim
    if cfg.pos_encoding == "rope" and cfg.enforce_fa2 and cfg.rotary_dim is not None:
        if cfg.rotary_dim > cfg.head_dim:
            fa2_notes.append(f"Set rotary_dim from {cfg.rotary_dim} down to head_dim={cfg.head_dim}.")
            cfg.rotary_dim = cfg.head_dim
        if cfg.rotary_dim % 16 != 0:
            new_rd = _nearest_multiple(cfg.rotary_dim, 16)
            new_rd = min(new_rd, cfg.head_dim)
            fa2_notes.append(f"Rounded rotary_dim to {new_rd} (KV-cache path prefers multiples of 16).")
            cfg.rotary_dim = new_rd

    # “Param coefficient” for the quadratic solver (used if not tying heads to layers)
    mlp_coeff = (3.0 * cfg.ffn_mult) if cfg.linear_type == "swiglu" else (2.0 * cfg.ffn_mult)

    best = None

    for L in range(cfg.min_layers, cfg.max_layers + 1):

        if cfg.tie_heads_to_layers:
            # Start from n_heads ≈ L, then make it legal for kernels and GQA
            step = cfg.round_heads_to if cfg.round_heads_to else 1
            H = _nearest_multiple(L, step)

            if cfg.n_kv_heads:
                # ensure divisibility by n_kv_heads (and keep close to L)
                m = _lcm(step, cfg.n_kv_heads)
                H = _nearest_multiple(L, m)

            D = H * cfg.head_dim
        else:
            # Depth/width from parameter budget (classic heuristic)
            D_cont = _solve_hidden_given_layers(cfg.target_params, cfg.vocab_size, cfg.context_length,
                                                L, mlp_coeff, cfg.pos_encoding)
            # round hidden and get heads from head_dim
            H = max(step if (step := cfg.round_heads_to) else 1, int(round(D_cont / cfg.head_dim)))
            H = _nearest_multiple(H, step)
            D = H * cfg.head_dim

        F = int(round(cfg.ffn_mult * D))
        params_total, params_emb, breakdown = _count_params(
            L, D, F, cfg.vocab_size, cfg.context_length, cfg.pos_encoding, cfg.linear_type
        )
        # Score: primary = parameter error; secondary = heads ~ layers closeness
        err = abs(params_total - cfg.target_params)
        closeness = abs(H - L) if cfg.tie_heads_to_layers else 0
        score = (err, closeness)

        cand = (score, (L, D, H, F, params_total, params_emb, breakdown))
        if best is None or cand[0] < best[0]:
            best = cand

    # Unpack best choice
    (_, (L, D, n_heads, F, params_total, params_emb, breakdown)) = best

    # FLOPs (B=1) for full sequence length (prefill)
    fwd_flops, bwd_flops, flops_breakdown = _estimate_flops(
        B=B_for_flops, T=cfg.context_length, L=L, D=D, F=F,
        linear_type=cfg.linear_type, include_vocab=include_vocab_in_flops, V=cfg.vocab_size
    )

    # KV heads for memory
    n_kv = cfg.n_kv_heads if cfg.n_kv_heads else n_heads

    # Memory estimates
    mem_fwd = _estimate_memory_forward_inference(
        params_total, L, cfg.context_length, n_kv_heads=n_kv, head_dim=cfg.head_dim,
        dtype_bytes=2, keep_master_fp32=True
    )
    mem_bwd = _estimate_memory_backward_training(
        params_total, B=1, T=cfg.context_length, L=L, D=D, dtype_bytes=2
    )

    details = {
        "attn_params_per_layer": breakdown["attn_per_layer"],
        "mlp_params_per_layer": breakdown["mlp_per_layer"],
        "norm_params_per_layer": breakdown["norm_per_layer"],
        "ratio_heads_over_layers": n_heads / max(1, L),
        "tie_heads_to_layers": cfg.tie_heads_to_layers,
        "n_kv_heads": n_kv,
        "head_dim": cfg.head_dim,
        "rotary_dim": cfg.rotary_dim if cfg.pos_encoding == "rope" else None,
        "fa2_notes": fa2_notes,
        "includes_vocab_in_flops": include_vocab_in_flops,
        "assumptions": "MHA (no biases), weight tying, bf16 activations; AdamW; checkpointed training"
    }

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
        details=details
    )

def pretty_print(design: LLMDesignOutputs) -> None:
    print("=== Recommended Architecture ===")
    print(f"Layers (L):         {design.n_layers}")
    print(f"Heads (N):          {design.n_heads}  (N/L = {design.details['ratio_heads_over_layers']:.3f})")
    print(f"Head dim (H):       {design.head_dim}")
    print(f"Hidden size (D):    {design.hidden_size}")
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
    print()
    if design.details.get("fa2_notes"):
        print("FA2 compliance notes:")
        for note in design.details["fa2_notes"]:
            print("  - " + note)

if __name__ == "__main__":
    model_targets = [
        ("gpt2-124m", 124_000_000),
        ("gpt2-355m", 355_000_000),
        ("gpt2-774m", 774_000_000),
        ("gpt2-1.5b", 1_500_000_000),
        ("custom-1.3b", 1_300_000_000),
        ("custom-2.7b", 2_700_000_000),
        ("custom-3.8b", 3_800_000_000),
        ("custom-7b", 7_000_000_000),
        ("custom-14b", 14_000_000_000),
        ("custom-32b", 32_000_000_000),
        ("custom-70b", 70_000_000_000),
    ]

    headers = [
        "model",
        "target_params",
        "params_total",
        "layers",
        "hidden",
        "heads",
        "head_dim",
        "ffn_dim",
        "kv_heads",
        "rotary_dim",
        "context",
        "linear_type",
    ]
    print("\t".join(headers))

    for model_name, params in model_targets:
        cfg = LLMDesignInputs(
            target_params=params,
            vocab_size=32_000,
            context_length=8192,
            linear_type="swiglu",
            pos_encoding="rope",
            tie_heads_to_layers=True,
        )
        design = recommend_llm_config(cfg, B_for_flops=1, include_vocab_in_flops=False)

        row = [
            model_name,
            str(params),
            str(design.params_total),
            str(design.n_layers),
            str(design.hidden_size),
            str(design.n_heads),
            str(design.head_dim),
            str(design.ffn_dim),
            str(design.details.get("n_kv_heads")),
            str(design.details.get("rotary_dim")),
            str(cfg.context_length),
            cfg.linear_type,
        ]
        print("\t".join(row))
