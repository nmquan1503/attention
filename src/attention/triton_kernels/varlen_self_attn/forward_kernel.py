import torch
import triton
import triton.language as tl

@triton.jit
def traditional_attention_kernel(
    q_ptr,              # (total_tokens, num_heads, dim)
    k_ptr,              # (total_tokens, num_heads, dim)
    v_ptr,              # (total_tokens, num_heads, dim)
    out_ptr,            # (total_tokens, num_heads, dim)
    cu_seqlens_ptr,     # (num_seqs + 1,)
    q_t_stride, q_h_stride, q_d_stride,
    k_t_stride, k_h_stride, k_d_stride,
    v_t_stride, v_h_stride, v_d_stride,
    out_t_stride, out_h_stride, out_d_stride,
    NUM_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    q_block_id = tl.program_id(2)

    seq_start = tl.load(cu_seqlens_ptr + seq_id)
    seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
    seq_len = seq_end - seq_start

    if q_block_id * BLOCK_Q >= seq_len:
        return

    offs_q = q_block_id * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < seq_len
    q_ids = seq_start + offs_q

    offs_d = tl.arange(0, D)

    q = tl.load(
        q_ptr + q_ids[:, None] * q_t_stride + head_id * q_h_stride + offs_d[None, :] * q_d_stride,
        mask=mask_q[:, None],
        other=0.0
    )

    row_max = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    for k_block_start in range(0, seq_len, BLOCK_K):
        offs_k = k_block_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < seq_len
        k_ids = seq_start + offs_k

        k = tl.load(
            k_ptr + k_ids[:, None] * k_t_stride + head_id * k_h_stride + offs_d[None, :] * k_d_stride,
            mask=mask_k[:, None],
            other=0.0
        )

        scores = tl.dot(q, tl.trans(k)) * SCALE

        if IS_CAUSAL:
            allowed = k_ids[None, :] <= q_ids[:, None]
        else:
            allowed = k_ids[None, :] >= 0

        scores = tl.where(allowed, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - new_max)

        acc *= alpha[:, None]
        row_sum *= alpha

        exp_scores = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(exp_scores, axis=1)

        v = tl.load(
            v_ptr + k_ids[:, None] * v_t_stride + head_id * v_h_stride + offs_d[None, :] * v_d_stride,
            mask=mask_k[:, None],
            other=0.0
        )

        acc += tl.dot(exp_scores, v)
        row_sum += block_sum
        row_max = new_max

    acc /= row_sum[:, None]

    tl.store(
        out_ptr + q_ids[:, None] * out_t_stride + head_id * out_h_stride + offs_d[None, :] * out_d_stride,
        acc,
        mask=mask_q[:, None]
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def pack_sequences_kernel(
    x_ptr,                  # (batch_size, seq_len, dim)
    packed_ptr,             # (total_tokens, dim)
    cu_seqlens_ptr,         # (batch_size + 1,)
    pack_offsets_ptr,       # (batch_size + 1,)
    x_b_stride, x_s_stride, x_d_stride,
    packed_t_stride, packed_d_stride,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, B

    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(pack_offsets_ptr + mid) <= block_id:
            lo = mid + 1
        else:
            hi = mid

    b = lo - 1
    t_block = block_id - tl.load(pack_offsets_ptr + b)

    seq_start = tl.load(cu_seqlens_ptr + b)
    seq_len = tl.load(cu_seqlens_ptr + b + 1) - seq_start

    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (t[:, None] < seq_len) & (d[None, :] < D)

    x_ptrs = (
        x_ptr
        + b * x_b_stride
        + t[:, None] * x_s_stride
        + d[None, :] * x_d_stride
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)

    packed_ptrs = (
        packed_ptr
        + (seq_start + t[:, None]) * packed_t_stride
        + d[None, :] * packed_d_stride
    )

    tl.store(packed_ptrs, x, mask=mask)



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def unpack_sequences_kernel(
    packed_ptr,             # (total_tokens, dim)
    x_ptr,                  # (batch_size, max_seq_len, dim)
    cu_seqlens_ptr,         # (batch_size + 1,)
    unpack_offsets_ptr,     # (batch_size + 1,)
    packed_t_stride, packed_d_stride,
    x_b_stride, x_s_stride, x_d_stride,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_id = tl.program_id(0)
    d_block_id = tl.program_id(1)

    lo, hi = 0, B
    while lo < hi:
        mid = (lo + hi) // 2

        if tl.load(unpack_offsets_ptr + mid) <= block_id:
            lo = mid + 1
        else:
            hi = mid

    b = lo - 1
    t_block = block_id - tl.load(unpack_offsets_ptr + b)

    seq_start = tl.load(cu_seqlens_ptr + b)
    seq_len = tl.load(cu_seqlens_ptr + b + 1) - seq_start

    offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (offs_t[:, None] < seq_len) & (offs_d[None, :] < D)

    packed_ptrs = (
        packed_ptr
        + (seq_start + offs_t[:, None]) * packed_t_stride
        + offs_d[None, :] * packed_d_stride
    )
    val = tl.load(packed_ptrs, mask=mask, other=0.0)

    x_ptrs = (
        x_ptr
        + b * x_b_stride
        + offs_t[:, None] * x_s_stride
        + offs_d[None, :] * x_d_stride
    )
    tl.store(x_ptrs, val, mask=mask)