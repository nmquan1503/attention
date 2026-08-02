import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 64}, num_warps=8),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ],
    key=["D", "BLOCK_T"],
)
@triton.jit
def pad_buffer_kernel(
    k_cache_ptr,                # (total_slots, num_heads, dim)
    v_cache_ptr,                # (total_slots, num_heads, dim)
    k_out_ptr,                  # (new_total_slots, num_heads, dim)
    v_out_ptr,                  # (new_total_slots, num_heads, dim)
    cu_seqlens_cache_ptr,       # (num_seqs + 1,)
    new_cu_seqlens_cache_ptr,   # (num_seqs + 1,)
    write_pos_ptr,              # (num_seqs,)
    pad_offsets_ptr,            # (num_seqs + 1,)
    k_cache_t_stride, k_cache_h_stride, k_cache_d_stride,
    k_out_t_stride, k_out_h_stride, k_out_d_stride,
    NUM_SEQS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    d_block_id = tl.program_id(2)

    lo, hi = 0, NUM_SEQS
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(pad_offsets_ptr + mid) <= token_block_id:
            lo = mid + 1
        else:
            hi = mid
    seq_id = lo - 1
    block_in_seq = token_block_id - tl.load(pad_offsets_ptr + seq_id)

    old_start = tl.load(cu_seqlens_cache_ptr + seq_id)
    old_end = tl.load(write_pos_ptr + seq_id)
    length = old_end - old_start
    new_start = tl.load(new_cu_seqlens_cache_ptr + seq_id)

    offs_t = block_in_seq * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < length
    offs_d = d_block_id * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask = mask_t[:, None] & mask_d[None, :]

    k = tl.load(
        k_cache_ptr + (old_start + offs_t[:, None]) * k_cache_t_stride
                     + head_id * k_cache_h_stride
                     + offs_d[None, :] * k_cache_d_stride,
        mask=mask, other=0.0,
    )
    tl.store(
        k_out_ptr + (new_start + offs_t[:, None]) * k_out_t_stride
                   + head_id * k_out_h_stride
                   + offs_d[None, :] * k_out_d_stride,
        k, mask=mask,
    )

    v = tl.load(
        v_cache_ptr + (old_start + offs_t[:, None]) * k_cache_t_stride
                     + head_id * k_cache_h_stride
                     + offs_d[None, :] * k_cache_d_stride,
        mask=mask, other=0.0,
    )
    tl.store(
        v_out_ptr + (new_start + offs_t[:, None]) * k_out_t_stride
                   + head_id * k_out_h_stride
                   + offs_d[None, :] * k_out_d_stride,
        v, mask=mask,
    )


@triton.jit
def traditional_decode_kernel(
    q_ptr,                      # (num_seqs, num_heads, dim)
    k_ptr,                      # (num_seqs, num_heads, dim)
    v_ptr,                      # (num_seqs, num_heads, dim)
    k_cache_ptr,                # (total_slots, num_heads, dim)
    v_cache_ptr,                # (total_slots, num_heads, dim)
    cu_seqlens_cache_ptr,       # (num_seqs + 1,)
    write_pos_ptr,              # (num_seqs,)
    mid_o_ptr,                  # (num_seqs * num_heads, num_chunks, dim)
    mid_logsumexp_ptr,          # (num_seqs * num_heads, num_chunks)
    q_t_stride, q_h_stride, q_d_stride,
    k_cache_t_stride, k_cache_h_stride, k_cache_d_stride,
    mid_o_g_stride, mid_o_c_stride, mid_o_d_stride,
    mid_logsumexp_g_stride, mid_logsumexp_c_stride,
    NUM_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    chunk_id = tl.program_id(2)
    group_id = seq_id * NUM_HEADS + head_id

    cache_start = tl.load(cu_seqlens_cache_ptr + seq_id)
    current_end = tl.load(write_pos_ptr + seq_id)

    chunk_start = cache_start + chunk_id * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, current_end)
    chunk_len = chunk_end - chunk_start

    if chunk_len < 0:
        return

    offs_d = tl.arange(0, D)

    q = tl.load(
        q_ptr + seq_id * q_t_stride + head_id * q_h_stride + offs_d[None, :] * q_d_stride
    )

    if chunk_len == CHUNK_SIZE:
        acc = tl.zeros((D,), dtype=tl.float32)
        m = tl.full((), -float("inf"), dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)
    else:
        k_new = tl.load(
            k_ptr + seq_id * q_t_stride + head_id * q_h_stride + offs_d[:, None] * q_d_stride
        )
        v_new = tl.load(
            v_ptr + seq_id * q_t_stride + head_id * q_h_stride + offs_d * q_d_stride
        )

        new_pos = current_end
        tl.store(k_cache_ptr + new_pos * k_cache_t_stride + head_id * k_cache_h_stride + offs_d * k_cache_d_stride,
                 tl.reshape(k_new, (D,)))
        tl.store(v_cache_ptr + new_pos * k_cache_t_stride + head_id * k_cache_h_stride + offs_d * k_cache_d_stride, v_new)

        m = tl.reshape(tl.dot(q, k_new), ()).to(tl.float32) * SCALE
        l = 1.0
        acc = v_new

        chunk_end = current_end + 1
        chunk_len = chunk_end - (cache_start + chunk_id * CHUNK_SIZE)

    offs_k_base = tl.arange(0, BLOCK_K)
    for block_start in range(0, chunk_len, BLOCK_K):
        offs_k = (cache_start + chunk_id * CHUNK_SIZE + block_start) + offs_k_base
        mask_k = offs_k < chunk_end

        k_loaded = tl.load(
            k_cache_ptr + offs_k[:, None] * k_cache_t_stride
                         + head_id * k_cache_h_stride
                         + offs_d[None, :] * k_cache_d_stride,
            mask=mask_k[:, None], other=0.0,
        )

        scores = tl.reshape(tl.dot(q, tl.trans(k_loaded)), (BLOCK_K,)) * SCALE

        block_max = tl.max(scores)
        new_max = tl.maximum(m, block_max)
        alpha = tl.exp(m - new_max)
        p = tl.exp(scores - new_max)

        v_loaded = tl.load(
            v_cache_ptr + offs_k[:, None] * k_cache_t_stride
                         + head_id * k_cache_h_stride
                         + offs_d[None, :] * k_cache_d_stride,
            mask=mask_k[:, None], other=0.0,
        )
        acc = acc * alpha + tl.reshape(tl.dot(p[None, :], v_loaded), (D,))
        l = l * alpha + tl.sum(p)
        m = new_max

    mid_o = acc / l
    tl.store(mid_o_ptr + group_id * mid_o_g_stride + chunk_id * mid_o_c_stride + offs_d * mid_o_d_stride, mid_o)
    tl.store(mid_logsumexp_ptr + group_id * mid_logsumexp_g_stride + chunk_id * mid_logsumexp_c_stride, m + tl.log(l))


@triton.jit
def reduce_kernel_traditional(
    mid_o_ptr,                  # (num_seqs * num_heads, num_chunks, dim)
    mid_logsumexp_ptr,          # (num_seqs * num_heads, num_chunks)
    cu_seqlens_cache_ptr,       # (num_seqs + 1,)
    write_pos_ptr,              # (num_seqs,)
    out_ptr,                    # (num_seqs, num_heads, dim)
    mid_o_g_stride, mid_o_c_stride, mid_o_d_stride,
    mid_logsumexp_g_stride, mid_logsumexp_c_stride,
    out_t_stride, out_h_stride, out_d_stride,
    NUM_HEADS: tl.constexpr,
    D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    group_id = seq_id * NUM_HEADS + head_id

    cache_start = tl.load(cu_seqlens_cache_ptr + seq_id)
    current_end = tl.load(write_pos_ptr + seq_id)
    cache_len = current_end - cache_start + 1
    num_chunks = tl.cdiv(cache_len, CHUNK_SIZE)

    offs_d = tl.arange(0, D)
    acc = tl.zeros((D,), dtype=tl.float32)
    m = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)

    for chunk_id in range(num_chunks):
        cur_logsumexp = tl.load(mid_logsumexp_ptr + group_id * mid_logsumexp_g_stride + chunk_id * mid_logsumexp_c_stride)
        cur_o = tl.load(mid_o_ptr + group_id * mid_o_g_stride + chunk_id * mid_o_c_stride + offs_d * mid_o_d_stride)
        new_m = tl.maximum(m, cur_logsumexp)
        alpha = tl.exp(m - new_m)
        weight = tl.exp(cur_logsumexp - new_m)
        acc = acc * alpha + cur_o * weight
        l = l * alpha + weight
        m = new_m

    out = acc / l
    tl.store(out_ptr + seq_id * out_t_stride + head_id * out_h_stride + offs_d * out_d_stride, out)
    tl.store(write_pos_ptr + seq_id, current_end + 1)