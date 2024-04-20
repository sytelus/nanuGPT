# Adopted from: https://colab.research.google.com/drive/1Zr53EaAi5LQueyhbZv4OMZrkaQGuOWe0?usp=sharing#scrollTo=PE-G5hNjl0dF

def get_num_params(d_model, n_layers, vocab_size, include_embedding=True, bias=False):
    bias = bias * 1

    wpe = seq_len * d_model
    # wte shared with head so don't count
    model_ln = d_model + d_model * bias
    head_params = d_model * vocab_size

    # atn_block
    qkv = d_model * 3 * d_model
    w_out = d_model ** 2
    ln = model_ln
    attn_params = qkv + w_out + ln

    # ff
    ff = d_model * d_model * 4
    ln = model_ln
    ff_params = ff * 2 + ln

    params = (attn_params + ff_params) * n_layers + model_ln + head_params
    if include_embedding:
        params += wpe
    return params

def get_activations_mem(d_model, n_layers, vocab_size, n_heads, bsz, seq_len):
    # activations
    layer_norm = bsz * seq_len * d_model * 4  # FP32

    tokens = bsz * seq_len * d_model   # Number of elements (comes up a lot)

    # attn block
    QKV = tokens * 2  # FP16
    QKT = 2 * tokens * 2  # FP16
    softmax = bsz * n_heads * seq_len ** 2 * 4  # FP32
    PV = softmax / 2 + tokens * 2  # FP16
    out_proj = tokens * 2  # FP16
    attn_act = layer_norm + QKV + QKT + softmax + PV + out_proj

    # FF block
    ff1 = tokens * 2  # FP16
    gelu = tokens * 4 * 2  # FP16
    ff2 = tokens * 4 * 2  # FP16
    ff_act = layer_norm + ff1 + gelu + ff2

    final_layer = tokens * 2  # FP16
    model_acts = layer_norm + (attn_act + ff_act) * n_layers + final_layer

    # extras
    cross_entropy1 = bsz * seq_len * vocab_size * 2  # FP16
    cross_entropy2 = cross_entropy1 * 2  # FP32?

    # bwds = cross_entropy2
    extras = cross_entropy1 + cross_entropy2
    mem = model_acts + extras
    return mem


def get_inputs_mem(bsz, seq_len):
    return bsz * seq_len * 8 * 2  # int64 input and targets


def get_mem_size(d_model, n_layers, vocab_size, n_heads, bsz, seq_len):
    MB = 1024 ** 2
    N = get_num_params(d_model, n_layers, vocab_size)
    B = n_layers * seq_len ** 2

    # all in bytes
    bufs = 4 * B
    weights = 4 * N
    grads = 4 * N
    adam = (4+4) * N
    logits = 2 * bsz * seq_len * vocab_size

    kernels = 8_519_680 # bytes, CuBLASS creates its own workspace, see appendix of post

    GB = 1024 ** 3
    MB = 1024 ** 2
    KB = 1024
    BYTES = 1
    unit = GB

    # Steady State
    model = bufs + weights
    inputs_mem = get_inputs_mem(bsz, seq_len)
    total_mem = model + grads + adam + kernels * 2 + inputs_mem
    print(f"Model memory (bytes): {total_mem:,.0f}", )

    # First call
    n_params = get_num_params(d_model, n_layers, vocab_size)
    inputs_mem = get_inputs_mem(bsz, seq_len)
    actv_mem = get_activations_mem(d_model, n_layers, vocab_size, n_heads, bsz, seq_len) / unit

    mem_pre_forward = ((kernels * 2) + inputs_mem +  weights + bufs) / unit
    print(f'{mem_pre_forward=:,.3f}')

    mem_post_forward = mem_pre_forward + actv_mem / unit
    print(f'{mem_post_forward=:,.3f}')

    mem_post_step = mem_pre_forward + (grads + adam) / unit
    print(f'{mem_post_step=:,.3f}')

    # Other calls
    mem_pre_forward = ((kernels * 2) + get_inputs_mem(bsz, seq_len) +  weights + bufs + adam + grads)  / unit
    print(f'{mem_pre_forward=:,.3f}')

    mem_post_forward = mem_pre_forward + actv_mem
    print(f'{mem_post_forward=:,.3f}')

    mem_post_step = mem_pre_forward
    print(f'{mem_post_step=:,.3f}')

    # peak estimation
    # http://erees.dev/transformer-memory/#total-peak-memory
    # additional part is the extra Nl
    mem_post_step_peak_allocated = mem_post_forward + (bsz * seq_len * vocab_size * 4) / unit
    print(f'{mem_post_step_peak_allocated=:,.3f}')


# GPT2
vocab_size = 50304
bsz = 12
seq_len = 1024
d_model, n_heads, n_layers = 768, 12, 12
get_mem_size(d_model=d_model, n_layers=n_layers, vocab_size=vocab_size, n_heads=n_heads, bsz=bsz, seq_len=seq_len)

d_model, n_heads, n_layers = 1024, 24, 16
get_mem_size(d_model=d_model, n_layers=n_layers, vocab_size=vocab_size, n_heads=n_heads, bsz=bsz, seq_len=seq_len)

bsz=9
d_model, n_heads, n_layers = 1280, 36, 20
get_mem_size(d_model=d_model, n_layers=n_layers, vocab_size=vocab_size, n_heads=n_heads, bsz=bsz, seq_len=seq_len)