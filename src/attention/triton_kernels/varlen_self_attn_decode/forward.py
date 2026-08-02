import torch
import triton
import math

from .forward_kernel import (
    pad_buffer_kernel,
    traditional_decode_kernel,
    reduce_kernel_traditional,
)


def pad_buffer(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_cache: torch.Tensor,
    write_pos: torch.Tensor,
    buffer_size: int,
):
    """
    Args:
        k_cache: (total_slots, num_heads, dim)
        v_cache: (total_slots, num_heads, dim)
        cu_seqlens_cache: (num_seqs + 1,)
        write_pos: (num_seqs,)
        buffer_size: int
    Returns:
        k_out: (new_total_slots, num_heads, dim)
        v_out: (new_total_slots, num_heads, dim)
        new_cu_seqlens_cache: (num_seqs + 1,)
        new_write_pos: (num_seqs,)
    """
    num_seqs = cu_seqlens_cache.numel() - 1
    _, num_heads, dim = k_cache.shape
    lengths = write_pos - cu_seqlens_cache[:-1]

    if not torch.any(lengths > 0):
        new_lengths = lengths + buffer_size
        new_cu_seqlens_cache = torch.zeros(num_seqs + 1, dtype=torch.int32, device=k_cache.device)
        new_cu_seqlens_cache[1:] = torch.cumsum(new_lengths, dim=0)
        new_total_slots = new_cu_seqlens_cache[-1].item()
        k_out = torch.empty(new_total_slots, num_heads, dim, device=k_cache.device, dtype=k_cache.dtype)
        v_out = torch.empty(new_total_slots, num_heads, dim, device=v_cache.device, dtype=v_cache.dtype)
        new_write_pos = new_cu_seqlens_cache[:-1].clone()
        return k_out, v_out, new_cu_seqlens_cache, new_write_pos

    avg_len = math.ceil(lengths[lengths > 0].float().mean().item())
    block_t = min(256, triton.next_power_of_2(avg_len))

    num_blocks_per_seq = triton.cdiv(lengths, block_t)
    pad_offsets = torch.zeros(num_seqs + 1, dtype=torch.int32, device=k_cache.device)
    pad_offsets[1:] = torch.cumsum(num_blocks_per_seq, dim=0)
    total_token_blocks = pad_offsets[-1].item()

    new_lengths = lengths + buffer_size
    new_cu_seqlens_cache = torch.zeros(num_seqs + 1, dtype=torch.int32, device=k_cache.device)
    new_cu_seqlens_cache[1:] = torch.cumsum(new_lengths, dim=0)
    new_total_slots = new_cu_seqlens_cache[-1].item()
    new_write_pos = new_cu_seqlens_cache[:-1] + lengths

    k_out = torch.empty(new_total_slots, num_heads, dim, device=k_cache.device, dtype=k_cache.dtype)
    v_out = torch.empty(new_total_slots, num_heads, dim, device=v_cache.device, dtype=v_cache.dtype)

    grid = lambda META: (
        total_token_blocks,
        num_heads,
        triton.cdiv(dim, META["BLOCK_D"]),
    )

    pad_buffer_kernel[grid](
        k_cache, v_cache,
        k_out, v_out,
        cu_seqlens_cache, new_cu_seqlens_cache, write_pos,
        pad_offsets,
        *k_cache.stride(),
        *k_out.stride(),
        NUM_SEQS=num_seqs,
        NUM_HEADS=num_heads,
        D=dim,
        BLOCK_T=block_t,
    )

    return k_out, v_out, new_cu_seqlens_cache, new_write_pos


def varlen_traditional_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_cache: torch.Tensor,
    write_pos: torch.Tensor,
):
    """
    Args:
        q: (num_seqs, num_heads, dim)
        k: (num_seqs, num_heads, dim)
        v: (num_seqs, num_heads, dim)
        scale: float
        k_cache: (total_slots, num_heads, dim)
        v_cache: (total_slots, num_heads, dim)
        cu_seqlens_cache: (num_seqs + 1,)
        write_pos: (num_seqs,)
    Returns:
        out: (num_seqs, num_heads, dim)
        k_cache: (total_slots, num_heads, dim)
        v_cache: (total_slots, num_heads, dim)
        write_pos: (num_seqs,)
    """
    num_seqs, num_heads, dim = q.shape
    num_groups = num_seqs * num_heads

    old_lengths = write_pos - cu_seqlens_cache[:-1]
    new_lengths = old_lengths + 1
    max_len = new_lengths.max().item()

    CHUNK_SIZE_MIN = 16
    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    chunk_size = max(
        CHUNK_SIZE_MIN,
        math.ceil(max_len / max(1, math.ceil(num_sms / num_groups))),
    )
    chunk_size = 1 << (chunk_size - 1).bit_length()
    num_chunks = math.ceil(max_len / chunk_size)

    MIN_BLOCK_K = 16
    MAX_BLOCK_K = 128
    dtype_size = q.element_size()
    SMEM_BUDGET = 64 * 1024
    max_block_k = (SMEM_BUDGET // dtype_size - 2 * dim) // (dim + 2)
    max_block_k = max(1, max_block_k)
    max_block_k = 1 << (max_block_k.bit_length() - 1)

    BLOCK_K = min(MAX_BLOCK_K, chunk_size, max_block_k)
    BLOCK_K = max(MIN_BLOCK_K, BLOCK_K)

    mid_o = torch.zeros(num_groups, num_chunks, dim, dtype=q.dtype, device=q.device)
    mid_logsumexp = torch.full((num_groups, num_chunks), float("-inf"), dtype=q.dtype, device=q.device)

    traditional_decode_kernel[(num_seqs, num_heads, num_chunks)](
        q, k, v,
        k_cache, v_cache,
        cu_seqlens_cache, write_pos,
        mid_o, mid_logsumexp,
        *q.stride(),
        *k_cache.stride(),
        *mid_o.stride(),
        *mid_logsumexp.stride(),
        NUM_HEADS=num_heads,
        SCALE=scale,
        D=dim,
        CHUNK_SIZE=chunk_size,
        BLOCK_K=BLOCK_K,
    )

    out = torch.empty_like(v)
    reduce_kernel_traditional[(num_seqs, num_heads)](
        mid_o, mid_logsumexp,
        cu_seqlens_cache, write_pos,
        out,
        *mid_o.stride(),
        *mid_logsumexp.stride(),
        *out.stride(),
        NUM_HEADS=num_heads,
        D=dim,
        CHUNK_SIZE=chunk_size,
    )

    return out, k_cache, v_cache, write_pos