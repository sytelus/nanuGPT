#!/usr/bin/env python3
# qwen25_fsdp_probe.py
import argparse, math, time, os, sys, warnings, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def assert_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this probe.")
    return torch.device("cuda")

def cuda_mb(x): return x / 1e6
def ms(ts): return ts * 1000.0

def seed_everything(seed=1234):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def print_table(rows, headers):
    colw = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def fmt_row(r): return " | ".join(str(v).rjust(colw[i]) for i, v in enumerate(r))
    sep = "-+-".join("-"*w for w in colw)
    print(fmt_row(headers)); print(sep)
    for r in rows: print(fmt_row(r))

# -----------------------------
# Qwen2.5-ish configs
# Numbers are approximate; sized for a realistic feel.
# ff_mult≈2.6667 for SwiGLU (common for Qwen family)
# vocab default kept moderate; override via --vocab if needed.
# -----------------------------
QWEN25_CONFIGS = {
    # name: (layers, hidden_size, heads, kv_heads, ff_mult, vocab)
    "qwen25-0_5b":  (24,   1024, 16,  4, 2.6667, 65536),
    "qwen25-1_8b":  (24,   2048, 16,  8, 2.6667, 65536),
    "qwen25-3b":    (36,   2560, 20, 10, 2.6667, 65536),
    "qwen25-7b":    (32,   4096, 32,  8, 2.6667, 65536),
    "qwen25-14b":   (48,   5120, 40, 10, 2.6667, 65536),
}

# -----------------------------
# RoPE helpers (simple, fast)
# -----------------------------
def build_rope_cache(seq_len, dim, base=10000.0, device="cuda"):
    # dim is per-head (Dh)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [T, dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(0)     # [1,1,T,dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin

def apply_rope(x, cos, sin):
    # x: [B, n_head, T, Dh]
    B, H, T, Dh = x.shape
    x1 = x[..., :Dh//2]
    x2 = x[..., Dh//2:]
    # rotate half
    x_rot = torch.cat([-x2, x1], dim=-1)
    out = (x * cos[..., :T, :]) + (x_rot * sin[..., :T, :])
    return out

# -----------------------------
# RMSNorm (epsilon outside mean)
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # x: (..., d)
        s = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return self.weight * x

# -----------------------------
# SwiGLU MLP
# -----------------------------
class SwiGLU(nn.Module):
    def __init__(self, d, mult=2.6667):
        super().__init__()
        hidden = int(mult * d)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# -----------------------------
# GQA Attention with RoPE + causal mask
# -----------------------------
class GQAAttn(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.n_kv = n_kv_heads
        self.dh = d // n_heads
        self.q_proj = nn.Linear(d, n_heads * self.dh, bias=False)
        self.k_proj = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.v_proj = nn.Linear(d, n_kv_heads * self.dh, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, rope_cos, rope_sin, attn_mask=None):
        # x: [B,T,D]
        B, T, D = x.shape
        H, Hkv, Dh = self.n_heads, self.n_kv, self.dh

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)      # [B,H,T,Dh]
        k = self.k_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)    # [B,Hkv,T,Dh]
        v = self.v_proj(x).view(B, T, Hkv, Dh).transpose(1, 2)    # [B,Hkv,T,Dh]

        # RoPE
        q = apply_rope(q, rope_cos, rope_sin)  # [B,H,T,Dh]
        k = apply_rope(k, rope_cos, rope_sin)  # [B,Hkv,T,Dh]

        # Expand kv across query groups if GQA
        if Hkv != H:
            # repeat each kv head to cover all q heads (grouped)
            rep = H // Hkv
            k = k.repeat_interleave(rep, dim=1)  # [B,H,T,Dh]
            v = v.repeat_interleave(rep, dim=1)  # [B,H,T,Dh]

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,T,T]
        if attn_mask is not None:
            att = att + attn_mask  # mask should be -inf on forbidden

        att = att.softmax(dim=-1)
        y = torch.matmul(att, v)                     # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(y)

# -----------------------------
# Qwen2.5-ish Block & Model
# -----------------------------
class Qwen25Block(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads, ff_mult):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = GQAAttn(d, n_heads, n_kv_heads)
        self.ln2 = RMSNorm(d)
        self.mlp = SwiGLU(d, mult=ff_mult)

    def forward(self, x, rope_cos, rope_sin, attn_mask):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Qwen25Like(nn.Module):
    def __init__(self, layers, d, n_heads, n_kv_heads, ff_mult, vocab):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([
            Qwen25Block(d, n_heads, n_kv_heads, ff_mult)
            for _ in range(layers)
        ])
        self.ln_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.vocab = vocab
        self.d = d
        self.layers_n = layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.ff_mult = ff_mult

    def forward(self, idx, rope_cos, rope_sin, attn_mask):
        # idx: [B,T]
        x = self.tok_emb(idx)
        for blk in self.layers:
            x = blk(x, rope_cos, rope_sin, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# FSDP plumbing (single GPU)
# -----------------------------
def fsdp_wrap(module, no_shard=True, use_orig_params=False, cpu_offload=False, amp_dtype=None):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy, MixedPrecision, CPUOffload
    )
    strat = ShardingStrategy.NO_SHARD if no_shard else ShardingStrategy.FULL_SHARD
    mp = MixedPrecision(param_dtype=amp_dtype) if amp_dtype is not None else None
    cpu = CPUOffload(offload_params=True) if cpu_offload else None
    m = FSDP(
        module,
        sharding_strategy=strat,
        use_orig_params=use_orig_params,
        mixed_precision=mp,
        cpu_offload=cpu,
    )
    return m

def init_dist_if_needed():
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

# -----------------------------
# Benchmarking
# -----------------------------
@torch.no_grad()
def build_causal_mask(T, device):
    # upper triangular mask filled with -inf above diagonal
    mask = torch.full((1, 1, T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

def run_scenario(name, make_model_opt, batcher, steps, amp_dtype, device, seq_len):
    # returns (fwd_mem_MB, bwd_mem_MB, fwd_ms, bwd_ms)
    fwd_mem, bwd_mem, fwd_t, bwd_t = [], [], [], []
    # small warmup to stabilize kernels
    warmup = min(2, steps//10) if steps >= 5 else 1
    total = warmup + steps

    for i in range(total):
        model, opt = make_model_opt()

        # build RoPE once per run (shared across steps for fair mem accounting)
        rope_cos, rope_sin = build_rope_cache(seq_len, model.d // model.n_heads, device=device)
        attn_mask = build_causal_mask(seq_len, device)

        x = batcher()  # fresh batch
        # AMP context
        if amp_dtype is None:
            amp_ctx = torch.autocast(device_type="cuda", enabled=False)
        else:
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        with amp_ctx:
            logits = model(x, rope_cos, rope_sin, attn_mask)
            # next-token CE loss
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                                   x[:, 1:].contiguous().view(-1))
        torch.cuda.synchronize()
        t1 = time.time()
        peak_fwd = torch.cuda.max_memory_allocated(device)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t2 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        peak_bwd = torch.cuda.max_memory_allocated(device)

        opt.step()
        opt.zero_grad(set_to_none=True)
        del model, opt, logits, loss, x, rope_cos, rope_sin, attn_mask
        torch.cuda.empty_cache()

        if i >= warmup:
            fwd_mem.append(cuda_mb(peak_fwd))
            bwd_mem.append(cuda_mb(peak_bwd))
            fwd_t.append(ms(t1 - t0))
            bwd_t.append(ms(t3 - t2))

    def avg(v): return sum(v)/max(1,len(v))
    return (avg(fwd_mem), avg(bwd_mem), avg(fwd_t), avg(bwd_t))

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-like FSDP single-GPU memory/time probe")
    parser.add_argument("--model", choices=list(QWEN25_CONFIGS.keys()), default="qwen25-3b")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10, help="averaging steps")
    parser.add_argument("--vocab", type=int, default=None, help="override vocab size")
    parser.add_argument("--bf16", action="store_true", help="use bf16 for AMP scenarios (recommended)")
    parser.add_argument("--fp16", action="store_true", help="use fp16 for AMP scenarios")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device_name, device_id = 'cpu', -1

    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ):
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", MASTER_ADDR="localhost", MASTER_PORT="12355")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            assert gpu_count-1 >= local_rank, f'LOCAL_RANK={local_rank} is greater than available GPUs={gpu_count}'
            torch.cuda.set_device(local_rank)
            device_name, device_id = f'cuda:{local_rank}', local_rank
        elif gpu_count == 1:
            torch.cuda.set_device(0)
            device_name,device_id = 'cuda:0', 0
        # for deterministic training, reset GPU after setting the device
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # let other values be controlled by env vars
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        torch.set_float32_matmul_precision('high')

    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://",
                            device_id=torch.device(device_name))  # type: ignore[arg-type]

    device = assert_cuda()
    seed_everything(args.seed)

    # pick AMP dtype for scenarios (1),(3),(4); scenario (2) disables AMP.
    amp_dtype_default = None
    if args.bf16:
        amp_dtype_default = torch.bfloat16
    elif args.fp16:
        amp_dtype_default = torch.float16
    else:
        # default to bf16 if available on device
        amp_dtype_default = torch.bfloat16

    layers, d, n_heads, n_kv, ff_mult, vocab_cfg = QWEN25_CONFIGS[args.model]
    vocab = args.vocab if args.vocab is not None else vocab_cfg

    print(f"\nQwen2.5-like config: {args.model}")
    print(f"layers={layers}, d={d}, heads={n_heads}, kv_heads={n_kv}, ff_mult={ff_mult}, vocab={vocab}")
    print(f"seq={args.seq}, batch={args.batch}, steps(avg)={args.steps}, amp_default={amp_dtype_default}\n")

    def make_batch():
        return torch.randint(0, vocab, (args.batch, args.seq), device=device)

    # ----- Scenario builders -----
    def mk_plain(amp_dtype):
        m = Qwen25Like(layers, d, n_heads, n_kv, ff_mult, vocab).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        # wrap model with lambda that ignores rope/attn_mask signature differences for timing code
        class Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.d = d; self.n_heads = n_heads
            def forward(self, x, rope_cos, rope_sin, attn_mask):
                return self.m(x, rope_cos, rope_sin, attn_mask)
        return Wrap(m).to(device), opt

    def mk_fsdp(use_orig_params, cpu_offload, amp_dtype):
        from torch.distributed.fsdp import MixedPrecision
        m = Qwen25Like(layers, d, n_heads, n_kv, ff_mult, vocab).to(device)
        m = fsdp_wrap(
            m,
            no_shard=True,
            use_orig_params=use_orig_params,
            cpu_offload=cpu_offload,
            amp_dtype=amp_dtype,
        )
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        # annotate for timing wrapper
        class Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.d = d; self.n_heads = n_heads
            def forward(self, x, rope_cos, rope_sin, attn_mask):
                return self.m(x, rope_cos, rope_sin, attn_mask)
        return Wrap(m).to(device), opt

    # ----- Run all 4 scenarios -----
    rows = []
    headers = [
        "Scenario",
        "FWD Mem (MB)",
        "BWD Mem (MB)",
        "FWD Time (ms)",
        "BWD Time (ms)",
    ]

    # (1) Plain, AMP ON
    res1 = run_scenario(
        "plain_amp_on",
        make_model_opt=lambda: mk_plain(amp_dtype_default),
        batcher=make_batch,
        steps=args.steps,
        amp_dtype=amp_dtype_default,
        device=device,
        seq_len=args.seq,
    )
    rows.append(("Plain + AMP on", f"{res1[0]:.1f}", f"{res1[1]:.1f}", f"{res1[2]:.2f}", f"{res1[3]:.2f}"))

    # (2) Plain, AMP OFF
    res2 = run_scenario(
        "plain_amp_off",
        make_model_opt=lambda: mk_plain(None),
        batcher=make_batch,
        steps=args.steps,
        amp_dtype=None,   # disable AMP
        device=device,
        seq_len=args.seq,
    )
    rows.append(("Plain + AMP off", f"{res2[0]:.1f}", f"{res2[1]:.1f}", f"{res2[2]:.2f}", f"{res2[3]:.2f}"))

    # (3) FSDP NO_SHARD, use_orig_params=False
    res3 = run_scenario(
        "fsdp_noshard_flat",
        make_model_opt=lambda: mk_fsdp(use_orig_params=False, cpu_offload=False, amp_dtype=amp_dtype_default),
        batcher=make_batch,
        steps=args.steps,
        amp_dtype=amp_dtype_default,
        device=device,
        seq_len=args.seq,
    )
    rows.append(("FSDP NO_SHARD (orig_params=False)", f"{res3[0]:.1f}", f"{res3[1]:.1f}", f"{res3[2]:.2f}", f"{res3[3]:.2f}"))

    # (4) FSDP NO_SHARD, use_orig_params=True, CPU offload (params)
    res4 = run_scenario(
        "fsdp_noshard_orig_offload",
        make_model_opt=lambda: mk_fsdp(use_orig_params=True, cpu_offload=True, amp_dtype=amp_dtype_default),
        batcher=make_batch,
        steps=args.steps,
        amp_dtype=amp_dtype_default,
        device=device,
        seq_len=args.seq,
    )
    rows.append(("FSDP NO_SHARD (orig_params=True, offload)", f"{res4[0]:.1f}", f"{res4[1]:.1f}", f"{res4[2]:.2f}", f"{res4[3]:.2f}"))

    print_table(rows, headers)
    print("\nNote:")
    print("• FWD/BWD memory is the average of CUDA peak allocations per phase.")
    print("• FWD/BWD time excludes the optimizer step; it measures forward (loss compute) and backward only.")
    print("• Scenario (4) offloads *parameters* to CPU; it will reduce VRAM at the cost of speed.")
    print("• If you see OOM, lower --batch or --seq, or choose a smaller --model preset.")

if __name__ == "__main__":
    main()
