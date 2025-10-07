#!/usr/bin/env python3
# bench_llm_cpu_gpu_zentorch.py
#
# Benchmark a small LLM-style block (Multi-Head Attention + FFN/SwiGLU) in three modes:
#   1) "cuda"     : All GPU using CUDA stack (optional torch.compile with inductor)
#   2) "zentorch" : All CPU using ZenDNN via zentorch backend (BF16 autocast)
#   3) "cpu"      : All CPU using generic PyTorch (no zentorch)
#
# Prints env info, params, model size, tokens/s, ms/iter, and attention/MLP breakdown.

import os, math, time, platform, sys
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CONFIG — tweak here
# -----------------------------
MODE = "zentorch"         # "cuda" | "zentorch" | "cpu"
USE_SWIGLU = True         # True: SwiGLU MLP; False: plain FFN (SiLU)
LAYERS = 4                # stack a few blocks to be realistic
D_MODEL = 2048
N_HEADS = 16
# For FFN size: typical ratios ~ 4x for FFN; for SwiGLU often ~ (2/3)*4x per gate/up then down
D_FF = 4 * D_MODEL        # you can set a specific number (e.g., 11008 for 4096 models)

BATCH = 4                 # batch size
SEQ = 1024                # sequence length (tokens)
WARMUP = 8
ITERS = 32

# CUDA tuning (used only when MODE=="cuda")
CUDA_USE_BF16_IF_AVAILABLE = True   # use BF16 autocast if GPU supports it, else FP16
CUDA_COMPILE_WITH_INDUCTOR = False  # optional: torch.compile(..., backend="inductor")

# CPU tuning
CPU_BF16_WEIGHTS = True   # keep weights in BF16 (accumulation stays FP32 via autocast)
PIN_THREADS = True        # honors torch.set_num_threads; for EPYC set OMP envs externally

# Reproducibility
torch.manual_seed(1234)

# -----------------------------
# Utilities
# -----------------------------
def pretty_bytes(n):
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"

def param_count_and_size_bytes(model, dtype=torch.float32):
    n = sum(p.numel() for p in model.parameters())
    # assume storage in given dtype (rough size estimate)
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return n, n * bytes_per

class Timer:
    """Small helper that works for CPU or CUDA."""
    def __init__(self, device_type: str):
        self.device_type = device_type
        if device_type == "cuda":
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.stop_ev  = torch.cuda.Event(enable_timing=True)
        self.reset()

    def reset(self):
        self.total_ms = 0.0

    def __enter__(self):
        if self.device_type == "cuda":
            torch.cuda.synchronize()
            self.start_ev.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device_type == "cuda":
            self.stop_ev.record()
            torch.cuda.synchronize()
            self.total_ms += self.start_ev.elapsed_time(self.stop_ev)
        else:
            self.total_ms += (time.perf_counter() - self._t0) * 1e3

# -----------------------------
# Model components
# -----------------------------
class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)                         # [B,T,3D]
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.dk).transpose(1, 2)  # [B,H,T,dk]
        k = k.view(B, T, self.n_heads, self.dk).transpose(1, 2)  # [B,H,T,dk]
        v = v.view(B, T, self.n_heads, self.dk).transpose(1, 2)  # [B,H,T,dk]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))  # [B,H,T,T]
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)     # [B,T,D]
        return self.proj(out)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=True)
        self.act = nn.SiLU()
        self.down = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, x):
        return self.down(self.act(self.up(x)))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=True)
        self.up   = nn.Linear(d_model, d_ff, bias=True)
        self.down = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, x):
        g = F.silu(self.gate(x))
        u = self.up(x)
        return self.down(g * u)

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, swiglu=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff) if swiglu else FFN(d_model, d_ff)

    def forward(self, x, timers=None, device_type="cpu"):
        # Attention
        t_attn = timers["attn"]
        with t_attn:
            a = self.attn(self.ln1(x))
        x = x + a

        # MLP
        t_mlp = timers["mlp"]
        with t_mlp:
            m = self.mlp(self.ln2(x))
        x = x + m
        return x

class TinyLLM(nn.Module):
    def __init__(self, layers, d_model, n_heads, d_ff, swiglu=True):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model, bias=False)  # dummy "input proj"
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, swiglu=swiglu) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model, bias=False)   # dummy "lm head"
    def forward(self, x, timers=None, device_type="cpu"):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, timers=timers, device_type=device_type)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# Build according to MODE
# -----------------------------
def maybe_import_zentorch():
    try:
        import zentorch  # noqa: F401
        return True
    except Exception as e:
        print("[WARN] zentorch import failed:", e)
        return False

def build_model_and_inputs(mode: str):
    device = "cpu"
    dtype_for_size = torch.bfloat16 if (mode in ("zentorch","cpu") and CPU_BF16_WEIGHTS) else torch.float32

    model = TinyLLM(LAYERS, D_MODEL, N_HEADS, D_FF, swiglu=USE_SWIGLU)

    if mode == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        device = "cuda"
        model = model.to(device)

        # choose CUDA autocast dtype
        use_bf16 = CUDA_USE_BF16_IF_AVAILABLE and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        compile_backend = "inductor" if CUDA_COMPILE_WITH_INDUCTOR else None
        if compile_backend:
            model = torch.compile(model, backend=compile_backend)

        x = torch.randn(BATCH, SEQ, D_MODEL, device=device)

        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        return model, x, device, "cuda", ctx, dtype_for_size

    elif mode == "zentorch":
        ok = maybe_import_zentorch()
        if not ok:
            raise RuntimeError("zentorch not installed — run: pip install zentorch")
        # keep weights BF16 if requested (lighter bandwidth)
        if CPU_BF16_WEIGHTS:
            model = model.to(torch.bfloat16)
        # route to ZenDNN
        model = torch.compile(model, backend="zentorch")
        x = torch.randn(BATCH, SEQ, D_MODEL, device="cpu")
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return model, x, "cpu", "cpu", ctx, dtype_for_size

    elif mode == "cpu":
        # generic PyTorch on CPU, float32 weights by default
        if PIN_THREADS:
            torch.set_num_threads(max(1, torch.get_num_threads()))
        x = torch.randn(BATCH, SEQ, D_MODEL, device="cpu")
        # We'll still try BF16 autocast for apples-to-apples numeric format
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return model, x, "cpu", "cpu", ctx, dtype_for_size

    else:
        raise ValueError(f"Unknown MODE {mode}")

# -----------------------------
# Benchmark
# -----------------------------
def main():
    print("== Config ==")
    print(f"MODE={MODE}  layers={LAYERS}  d_model={D_MODEL}  n_heads={N_HEADS}  d_ff={D_FF}  swiglu={USE_SWIGLU}")
    print(f"BATCH={BATCH}  SEQ={SEQ}  WARMUP={WARMUP}  ITERS={ITERS}")
    print(f"PyTorch={torch.__version__}  CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device={torch.cuda.get_device_name(0)}  BF16 supported={torch.cuda.is_bf16_supported()}")
    print(f"CPU={platform.processor()}  num_threads={torch.get_num_threads()}  Python={platform.python_version()}")

    model, x, device, device_type, amp_ctx, dtype_for_size = build_model_and_inputs(MODE)

    # Count params / model "size" assuming current storage dtype
    n_params, size_bytes = param_count_and_size_bytes(model, dtype=dtype_for_size)
    print(f"Params={n_params:,}  (approx storage {pretty_bytes(size_bytes)} @ {str(dtype_for_size).split('.')[-1]})")

    # Timers (per-forward)
    t_total = Timer(device_type)
    t_attn  = Timer(device_type)
    t_mlp   = Timer(device_type)
    timers = {"attn": t_attn, "mlp": t_mlp}

    # Warmup
    with torch.inference_mode(), amp_ctx:
        for _ in range(WARMUP):
            _ = model(x, timers=timers, device_type=device_type)
            if device_type == "cuda":
                torch.cuda.synchronize()

    # Benchmark
    t_total.reset(); t_attn.reset(); t_mlp.reset()
    t0 = time.perf_counter()
    with torch.inference_mode(), amp_ctx:
        for _ in range(ITERS):
            with t_total:
                y = model(x, timers=timers, device_type=device_type)
            if device_type == "cuda":
                torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1e3
    ms_per_iter = wall_ms / ITERS
    toks = BATCH * SEQ * ITERS
    toks_per_s = toks / ((t1 - t0) + 1e-12)

    print("\n== Results ==")
    print(f"Total wall: {wall_ms:,.2f} ms  |  {ms_per_iter:,.2f} ms/iter  |  throughput: {toks_per_s:,.0f} tokens/s")
    print(f"Breakdown (sum over {ITERS} iters):")
    print(f"  attention: {t_attn.total_ms:,.2f} ms  ({t_attn.total_ms/ITERS:.2f} ms/iter)")
    print(f"  MLP      : {t_mlp.total_ms:,.2f} ms  ({t_mlp.total_ms/ITERS:.2f} ms/iter)")

    # quick numerical check to keep the compiler honest
    checksum = float(y.abs().mean())
    print(f"\nSanity: output mean(abs) = {checksum:.6f}")

if __name__ == "__main__":
    main()
