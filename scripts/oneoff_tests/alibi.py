import math
import torch

def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def buffered_future_mask(tensor, actual_context_len, alibi):
    _future_mask = torch.empty(0)
    dim = tensor.size(1)
    # _future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
    if (
        _future_mask.size(0) == 0
        or (not _future_mask.device == tensor.device)
        or _future_mask.size(1) < actual_context_len
    ):
        _future_mask = torch.triu(
            fill_with_neg_inf(torch.zeros([actual_context_len, actual_context_len])), 1
        )
        _future_mask = _future_mask.unsqueeze(0) + alibi
    _future_mask = _future_mask.to(tensor)
    return _future_mask[:tensor.shape[0]*args.decoder_attention_heads, :dim, :dim]

attn_heads = 2
context_len = 4
actual_context_len = 2

slopes = torch.Tensor(get_slopes(attn_heads))
print(slopes)
#In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
#If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
#This works because the softmax operation is invariant to translation, and our bias functions are always linear.
factors = torch.arange(context_len).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
alibi = slopes.unsqueeze(1).unsqueeze(1) * factors
print(alibi)
alibi = alibi.view(attn_heads, 1, context_len)
alibi = alibi.repeat(actual_context_len, 1, 1)  # batch_size, 1, 1
print(alibi)

x = torch.randn(1, context_len, attn_heads)
attn_mask = buffered_future_mask(alibi, actual_context_len, alibi)
print(attn_mask)