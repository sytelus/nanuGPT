import math
import sys
from typing import Iterator, Tuple, Iterable

import torch

def stream_dim_bits_upper(
    tensors: Iterable["torch.Tensor"],
    rounding: str = "ceil",
    chunk_rows_hint: int = 2_000_000,
) -> Iterator[Tuple[int, ...]]:
    """
    Streaming per-dimension compressibility estimator (upper bound).

    For each incoming tensor X (same shape & dtype across the stream), yields a tuple
    (b0, b1, ..., b_{ndim-1}), where bj is an integer number of *bits per element*,
    estimating the ideal lossless rate if we compress along axis j using:
      - First element of each run along axis j encoded 'raw'
      - Subsequent elements encoded as XOR with the previous element (along axis j)
      - Coding with a zero-order (per-bit independent) model -> sums of binary entropies

    This yields an UPPER BOUND on the true optimal rate (ignores cross-bit dependencies).
    It is online: results are recomputed after each tensor, using all data seen so far.

    Memory: O(ndim * bitwidth). No gradients are created.

    Args:
        tensors: iterator of torch.Tensor with fixed shape/dtype
        rounding: 'ceil' | 'floor' | 'nearest' for the yielded integer bits/value
        chunk_rows_hint: controls internal chunking when unpacking bits to limit peak memory

    Yields:
        Tuple[int, ...]  # one integer (bits per element) per dimension
    """
    import numpy as np
    import torch

    def _binary_entropy(p: np.ndarray) -> np.ndarray:
        # p in [0,1]; returns H_b(p) elementwise, with safe handling at 0/1
        p = p.astype(np.float64, copy=False)
        out = np.zeros_like(p)
        m = (p > 0.0) & (p < 1.0)
        q = p[m]
        out[m] = -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
        return out

    def _sum_unpacked_bits(arr2d_u8: np.ndarray, nbits: int) -> np.ndarray:
        # arr2d_u8: shape (N, bytes_per_elem)
        if arr2d_u8.size == 0:
            return np.zeros(nbits, dtype=np.int64)
        # Choose chunk rows so that each chunk is ~chunk_rows_hint bits
        rows_per_chunk = max(1, chunk_rows_hint // max(nbits, 1))
        s = np.zeros(nbits, dtype=np.int64)
        for start in range(0, arr2d_u8.shape[0], rows_per_chunk):
            chunk = arr2d_u8[start:start + rows_per_chunk]
            bits = np.unpackbits(chunk, axis=1)      # (rows, nbits)
            s += bits.sum(axis=0, dtype=np.int64)    # per-bit ones-count
        return s

    def _bytes_view_for_dtype_torch(t: "torch.Tensor") -> Tuple["np.ndarray", int]:
        """
        Return (byte_view, bytes_per_elem), where byte_view is an np.uint8 array with shape
        (*t.shape, bytes_per_elem) that presents the *exact stored bytes* for t's dtype.

        Supports ints, float16/32/64, and bfloat16 (by taking the top 16 bits of float32).
        """
        tc = t.detach().contiguous().cpu()
        dt = tc.dtype
        a_np = tc.numpy()  # for bfloat16 in older NumPy this may upcast to float32

        # Special handling for bfloat16: construct the exact 16-bit pattern from float32
        if dt == torch.bfloat16:
            # The numeric value is exactly representable in float32; the bfloat16 bit pattern
            # equals the *top* 16 bits of the float32 pattern.
            a32 = tc.to(torch.float32).numpy()  # no rounding error for bf16 -> f32
            bytes32 = a32.view(np.uint8).reshape(*a32.shape, 4)
            byteorder = a32.dtype.byteorder
            is_le = (byteorder == '<') or (byteorder == '=' and sys.byteorder == 'little')
            bf_bytes = bytes32[..., 2:4] if is_le else bytes32[..., 0:2]
            return bf_bytes, 2

        # All other supported dtypes: view as bytes directly
        bytes_per_elem = a_np.dtype.itemsize
        byte_view = a_np.view(np.uint8).reshape(*a_np.shape, bytes_per_elem)
        return byte_view, bytes_per_elem

    first = True
    ones_first = ones_delta = n_first = n_delta = None  # initialized after first item
    last_shape = None
    bytes_per_elem = None
    ndim = None

    for x in tensors:
        with torch.no_grad():
            if first:
                last_shape = tuple(x.shape)
                ndim = len(last_shape)
                byte_view, bytes_per_elem = _bytes_view_for_dtype_torch(x)
                nbits = 8 * bytes_per_elem
                ones_first = np.zeros((ndim, nbits), dtype=np.int64)
                ones_delta = np.zeros((ndim, nbits), dtype=np.int64)
                n_first = np.zeros(ndim, dtype=np.int64)
                n_delta = np.zeros(ndim, dtype=np.int64)
                first = False
            else:
                if tuple(x.shape) != last_shape:
                    raise ValueError(
                        f"All tensors must have the same shape: "
                        f"got {tuple(x.shape)} vs {last_shape}"
                    )
                byte_view, bp2 = _bytes_view_for_dtype_torch(x)
                if bp2 != bytes_per_elem:
                    raise ValueError("All tensors must have the same dtype/byte-size.")
                nbits = 8 * bytes_per_elem

            # For each axis, update counters for:
            #  - first element of each run along that axis (raw)
            #  - XOR differences for the rest along that axis
            for axis in range(ndim):
                b = np.moveaxis(byte_view, axis, 0)  # shape: (S_axis, ..., bytes)
                S_axis = b.shape[0]
                rest = int(np.prod(b.shape[1:-1], dtype=np.int64)) if b.ndim > 2 else 1
                b2 = b.reshape(S_axis, rest, bytes_per_elem)

                # First element of each run (raw)
                first_bytes = b2[0]  # (rest, bytes)
                ones_first[axis] += _sum_unpacked_bits(first_bytes, nbits)
                n_first[axis] += first_bytes.shape[0]

                # XOR deltas for subsequent elements
                if S_axis > 1:
                    deltab = np.bitwise_xor(b2[1:], b2[:-1])       # (S_axis-1, rest, bytes)
                    del_flat = deltab.reshape(-1, bytes_per_elem)  # ((S_axis-1)*rest, bytes)
                    ones_delta[axis] += _sum_unpacked_bits(del_flat, nbits)
                    n_delta[axis] += del_flat.shape[0]

            # Compute per-axis average bits/element (upper bound) and yield
            out_bits = []
            for axis in range(ndim):
                total_elems = n_first[axis] + n_delta[axis]
                if total_elems == 0:
                    H_avg = 0.0
                else:
                    H_first = 0.0
                    H_delta = 0.0
                    if n_first[axis] > 0:
                        p1_first = ones_first[axis] / n_first[axis]
                        H_first = float(_binary_entropy(p1_first).sum())
                    if n_delta[axis] > 0:
                        p1_delta = ones_delta[axis] / n_delta[axis]
                        H_delta = float(_binary_entropy(p1_delta).sum())
                    # Average over all elements contributing along this axis
                    H_avg = (n_first[axis] * H_first + n_delta[axis] * H_delta) / total_elems

                if rounding == "ceil":
                    out_bits.append(int(math.ceil(H_avg - 1e-12)))
                elif rounding == "floor":
                    out_bits.append(int(math.floor(H_avg + 1e-12)))
                elif rounding == "nearest":
                    out_bits.append(int(round(H_avg)))
                else:
                    raise ValueError("rounding must be one of {'ceil', 'floor', 'nearest'}")

            yield tuple(out_bits)

if __name__ == "__main__":
    # Stream of (N, C, H, W) float32 tensors, e.g., batches from a dataloader
    def gen():
        for _ in range(5):
            t = torch.randn(4, 16, 32, 32, dtype=torch.float32)
            # zero out some values
            t = torch.where((t > 0.31) | (t < 0.3), torch.tensor(0.0, dtype=torch.float32), t)
            yield t

    for bits_per_dim in stream_dim_bits_upper(gen(), rounding="ceil"):
        print(bits_per_dim)  # -> e.g., (11, 12, 10, 10) meaning per-element bits along N,C,H,W