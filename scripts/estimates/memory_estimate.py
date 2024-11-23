# Adopted from: https://colab.research.google.com/drive/1Zr53EaAi5LQueyhbZv4OMZrkaQGuOWe0?usp=sharing#scrollTo=PE-G5hNjl0dF
# Blog post: https://erees.dev/transformer-memory/

def get_num_params(d_model:int, n_layers:int, vocab_size:int, include_embedding, layer_norm_bias:bool, tied_emb:bool):
    wpe = seq_len * d_model # possitional embedding

    linear_head = d_model * vocab_size   # final projection to vocab size layer
    wte = linear_head if not tied_emb else 0 # token embedding

    # layer norm has one param for each feature
    layer_norm = d_model + (d_model if layer_norm_bias else 0)


    # atn_block
    qkv = d_model * 3 * d_model # matrix that project q,k,v first
    w_out = d_model * d_model # weight matrix that project the output of the attention block
    attn_params = qkv + w_out + layer_norm*2  # layer norm is applied before and after attention

    # ff
    ff1 = d_model * (d_model * 4)
    ff2 = d_model * (d_model * 4)
    ff_params = ff1 + ff2

    params = (attn_params + ff_params) * n_layers + layer_norm + linear_head # final layer norm and linear head
    if include_embedding:
        params += wpe + wte
    return params

def get_activations_mem(d_model, n_layers, vocab_size, n_heads, bsz, seq_len,
                        ff_factor=4,
                        layer_norm_size=4, attn_fp_size=2, softmax_fp_size=4,
                        ff_fp_size=2, loss_fp_size=2):
    # activations
    tokens = bsz * seq_len * d_model  # Number of elements (comes up a lot)

    layer_norm = tokens * layer_norm_size # layer norms are usually kept at FP32


    # attn block - these activations for individual layers are kept live so grads can be computed on backward pass
    proj = bsz * seq_len * d_model * 3 * attn_fp_size # Q, K, V
    attn_product = bsz * n_heads * seq_len * seq_len * attn_fp_size # QK^T
    softmax = bsz * n_heads * seq_len * seq_len * softmax_fp_size
    val_product = bsz * seq_len * d_model * attn_fp_size # softmax(QK^T)V
    out_proj = tokens * attn_fp_size
    attn_act = layer_norm + proj + attn_product + softmax + val_product + out_proj

    # FF block
    ff1 = tokens * ff_fp_size  # FP16
    gelu = tokens * ff_factor * ff_fp_size  # FP16
    ff2 = tokens * ff_factor * ff_fp_size  # FP16
    ff_act = layer_norm + ff1 + gelu + ff2

    final_layer = tokens * ff_fp_size  # FP16
    model_acts = layer_norm + (attn_act + ff_act) * n_layers + final_layer

    # extras
    cross_entropy1 = bsz * seq_len * vocab_size * loss_fp_size
    # Pytorch stores FP32 version as well?
    cross_entropy2 = bsz * seq_len * vocab_size * 4

    # bwds = cross_entropy2
    extras = cross_entropy1 + cross_entropy2
    mem = model_acts + extras
    return mem


def get_inputs_mem(bsz, seq_len, token_size):
    return bsz * seq_len * token_size * 2  # int64 input and targets


def get_mem_size(d_model, n_layers, vocab_size, n_heads, bsz, seq_len):
    MB = 1024 ** 2
    N = get_num_params(d_model, n_layers, vocab_size, True, True, True)
    B = n_layers * seq_len ** 2
    token_size = 4  #32-bit token

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

    # # Steady State
    # model = bufs + weights
    # inputs_mem = get_inputs_mem(bsz, seq_len)
    # total_mem = model + grads + adam + kernels * 2 + inputs_mem
    # print(f"Model memory (bytes): {total_mem:,.0f}", )

    # First call
    n_params = get_num_params(d_model, n_layers, vocab_size, True, True, True)
    inputs_mem = get_inputs_mem(bsz, seq_len, token_size)
    actv_mem = get_activations_mem(d_model, n_layers, vocab_size, n_heads, bsz, seq_len)

    # mem_pre_forward = ((kernels * 2) + inputs_mem +  weights + bufs) / unit
    # print(f'{mem_pre_forward=:,.3f}')

    # mem_post_forward = mem_pre_forward + actv_mem / unit
    # print(f'{mem_post_forward=:,.3f}')

    # mem_post_step = mem_pre_forward + (grads + adam) / unit
    # print(f'{mem_post_step=:,.3f}')

    # # Other calls
    # mem_pre_forward = ((kernels * 2) + get_inputs_mem(bsz, seq_len) +  weights + bufs + adam + grads)  / unit
    # print(f'{mem_pre_forward=:,.3f}')

    # mem_post_forward = mem_pre_forward + actv_mem
    # print(f'{mem_post_forward=:,.3f}')

    # mem_post_step = mem_pre_forward
    # print(f'{mem_post_step=:,.3f}')

    # peak estimation
    # http://erees.dev/transformer-memory/#total-peak-memory
    # additional part is the extra Nl
    # mem_post_step_peak_allocated = mem_post_forward + (bsz * seq_len * vocab_size * 4) / unit
    extra = (bsz * seq_len * vocab_size * 4) # 32-bit extra
    mem_post_step_peak_allocated = kernels*2 + bufs + inputs_mem + n_params*4 + actv_mem + logits + grads + adam + extra

    print(f'{mem_post_step_peak_allocated}')


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