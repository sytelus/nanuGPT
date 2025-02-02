# from https://raw.githubusercontent.com/KellerJordan/modded-nanogpt/09a49d4af4804af92d14216b43136f5510a8fba8/train_gpt2.py
# altered to use OpenWebText
# Test commandline:
# torchrun --nproc_per_node=1 --standalone keller_train_gpt2_adamw.py --batch_size 4 --num_iterations 6 --val_loss_every 2 --total_batch_size 4096 --enable_wandb 0
# python keller_train_gpt2_muon.py --batch_size 4 --num_iterations 6 --val_loss_every 2 --total_batch_size 4096 --enable_wandb 0

import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanugpt.common import setup_logger

with open(sys.argv[0]) as f:
    code = f.read()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :] # type: ignore

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying removed so they can be optimized at different rates
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

class DistributedDataLoader:
    def __init__(self, filepath, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        print0(f"DataLoader: loading data from {filepath}")
        if ddp_world_size > 1:
            self.tokens = np.array(np.memmap(filepath, dtype=np.uint16, mode="r"))
        else:
            self.tokens = np.memmap(filepath, dtype=np.uint16, mode="r")
        print0(f"DataLoader: loaded {len(self.tokens):,} tokens")
        self.ntok_total = len(self.tokens)

        assert self.ntok_total > process_rank*B*T + B*T, f"dataset is too small for this process rank: {self.ntok_total} <= {process_rank*B*T + B*T + 1}, ntok_total={self.ntok_total}, process_rank={process_rank}, B={B}, T={T}"
        print0(f"DataLoader: total number of tokens: {self.ntok_total:,}")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1): # type: ignore
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[torch.Tensor] = [*params]
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[torch.Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[torch.Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(0) / p_world.size(1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                if torch.distributed.is_initialized():
                    handle = torch.distributed.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}") # type: ignore

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="$DATA_ROOT/tokenized/openwebtext/tiktoken/train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="$DATA_ROOT/tokenized/openwebtext/tiktoken/val.bin", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="$OUT_DIR", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="d12", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=64, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=9536, help="number of iterations to run")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0018, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=256, help="learning rate warmup iterations")
    parser.add_argument("--cooldown_frac", type=float, default=0.4, help="learning rate warmdown iterations")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=250, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=200, help="how many batches of val to average?")
    parser.add_argument("--save_every", type=int, default=5000, help="every how many steps to save the checkpoint")
    parser.add_argument("--enable_wandb", type=int, default=1, help="disabled if 0")

    parser.add_argument("--run_id", type=str, default=time.strftime("%Y%m%d-%H%M%S"), help="unique identifier for this run")
    args = parser.parse_args()

    run_id = args.run_id

    args.input_bin = os.path.expandvars(args.input_bin)
    args.input_val_bin = os.path.expandvars(args.input_val_bin)
    args.output_dir = os.path.join(os.path.expandvars(args.output_dir), run_id)

    logger = setup_logger(config={
            "logging":{
                    "project_name": os.getenv("JOB_NAME", "keller_train_gpt2"),
                    "run_name": run_id,
                    "enable_wandb": args.enable_wandb != 0,
                    "log_dir": args.output_dir,
                    "log_filename": "log.txt",
                    "summaries_filename": "summary.txt",
                    "allow_overwrite_log": True,
                    "metrics_type": "classification",
                    "summaries_stdout": True,
                },
    })

    # convert args to dict and log
    logger.info(vars(args))

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.model in {"d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

    ddp_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    ddp_rank = int(os.environ.get('RANK', '0'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        assert gpu_count-1 >= ddp_local_rank, f'LOCAL_RANK={ddp_local_rank} is greater than available GPUs={gpu_count}'
        torch.cuda.set_device(ddp_local_rank)
        device_name = f'cuda:{ddp_local_rank}'
        device_id = ddp_local_rank
    elif gpu_count == 1:
        torch.cuda.set_device(0)
        device_name = 'cuda:0'
        device_id = 0
    else:
        raise ValueError('No GPU found. Set device_type=cpu.')

    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = 0 # each process gets the exact same seed
    print(f"device_name: {device_name}, ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}")

    use_ddp = ddp_world_size > 1
    if use_ddp:
        init_process_group(backend='nccl', device_id=torch.device(device_name))

    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size == tokens_per_fwdbwd

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) # type: ignore

    # init the model from scratch
    model_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(model)
    print0("compiling done.")

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    # here we wrap model into DDP container
    if use_ddp:
        model = DDP(model, device_ids=[device_id])
        raw_model:GPT = model.module # always contains the "raw" unwrapped model
    else:
        raw_model:GPT = model # type: ignore

    # collect the parameters to optimize
    hidden_matrix_params = [p for p in raw_model.transformer.h.parameters() if p.ndim == 2]
    embed_params = [raw_model.transformer.wte.weight]
    scalar_params = [p for p in raw_model.parameters() if p.ndim < 2]
    head_params = [raw_model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=ddp_rank, world_size=ddp_world_size)
    optimizers = [optimizer1, optimizer2]

    # learning rate schedule: stable then decay
    def get_lr(it: int):
        t = 1 - it / args.num_iterations # time remaining in training
        assert 1 >= t >= 0
        w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
        return w * 1.0 + (1 - w) * 0.1
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    timings = []
    train_tokens = 0
    train_start_time = time.time()
    train_time_hr = 0.0
    for step in range(args.num_iterations + 1):
        metrics = {}
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        val_time = 0.0
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            torch.cuda.synchronize()
            val_start = time.time()
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss
                val_loss /= args.val_max_steps
            if use_ddp:
                torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG)
            val_time = time.time() - val_start
            metrics.update({
                "train/step": step,
                "val/step": step,
                "val/loss": val_loss.item(),
                "val/time_s": val_time,
            })
            torch.cuda.synchronize()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            logger.info(metrics)
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        loss.backward()
        for p in model.parameters():
            p.grad = p.grad / (p.grad.norm() + 1e-6) # type: ignore

        # determine and set the learning rate for this iteration
        # momentum warmup for Muon
        frac = min(step / 300, 1)
        for group in optimizer2.param_groups:
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        model.zero_grad(set_to_none=True)

        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        train_tokens += ddp_world_size * B * T
        tokens_per_second = ddp_world_size * B * T / (t1-t0)
        lossf = loss.item() # keep track of the mean loss

        train_time_hr += (t1-t0-val_time) / 3600.0
        metrics.update({
            "train/loss": lossf,
            "train/token_per_sec": tokens_per_second,
            "train/step_interval": t1-t0-val_time,
            "train/elapsed_hr": (t1-train_start_time) / 3600.0,
            "train/train_time_hr": train_time_hr,
            "train/tokens": train_tokens,
            "train/lr": lr,
        })

        logger.info(metrics)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0-val_time)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]

    logger.info({
        "final_iters_avg (usec)": np.mean(timings),
        "final_iters_stddev (usec)": np.std(timings),
        "pick_memory_consumption (MiB)": torch.cuda.max_memory_allocated() // 1024 // 1024,
    })

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(model=raw_model.state_dict(), code=code, args=args.__dict__)
        torch.save(log, os.path.join(args.output_dir, 'final.pt'))

    # -------------------------------------------------------------------------
    # clean up nice
if use_ddp:
    destroy_process_group()