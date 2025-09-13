# repro script for the issue: https://discuss.pytorch.org/t/coordinate-descent-tuning-errors-out-with-torch-acceleratorerror-cuda-error-invalid-argument/223000

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel

class Block(nn.Module):
    def __init__(self, d_model=768, n_head=12, p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.p = p
    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab=50000, d_model=768, n_head=12, n_layer=12, seq_len=1024, p=0.1):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.h = nn.ModuleList([Block(d_model, n_head, p) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.wte.weight
    def forward(self, idx, y):
        B, S = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(S, device=idx.device))
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        preds = logits.view(-1, logits.size(-1))
        target = y.view(-1)
        loss = F.cross_entropy(preds, target, ignore_index=-1)
        # to get rid of error, use below instead
        # correct = (preds.argmax(dim=-1) == target).sum().detach().item()
        correct = (preds.argmax(dim=-1) == target).sum()
        return loss, correct

def main():
    from torch._dynamo import config as dconfig
    dconfig.capture_scalar_outputs = True
    sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    device = "cuda"
    B, S, D, C = 60, 1024, 768, 50000
    idx = torch.randint(0, C, (B, S), device=device)
    y = torch.randint(0, C, (B, S), device=device)
    model = TinyTransformer(vocab=C, d_model=D, n_head=12, n_layer=12, seq_len=S, p=0.1).to(device).half()
    model.train()
    model = torch.compile(model, fullgraph=True)
    loss, correct = model(idx, y)
    print(float(loss), correct)
    (loss + 0.0 * correct.float()).backward()
    torch.cuda.synchronize()
    print("ok")

if __name__ == "__main__":
    main()
