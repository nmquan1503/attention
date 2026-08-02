import torch
import triton
import math

from .forward_kernel import (
    traditional_attention_kernel,
    pack_sequences_kernel,
    unpack_sequences_kernel,
)


def _traditional_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """
    Args:
        q: (total_tokens, num_heads, dim)
        k: (total_tokens, num_heads, dim)
        v: (total_tokens, num_heads, dim)
        cu_seqlens: (num_seqs + 1,)
        max_seqlen: int
        scale: float
        is_causal: bool
    Returns:
        out: (total_tokens, num_heads, dim)
    """
    total_tokens, num_heads, dim = q.shape
    num_seqs = cu_seqlens.numel() - 1
    num_groups = num_seqs * num_heads

    out = torch.empty_like(q)

    SMEM_BUDGET = 32 * 1024
    MIN_BLOCK_Q = 8
    MIN_BLOCK_K = 16
    MAX_BLOCK_Q = min(128, triton.next_power_of_2(max_seqlen))
    MAX_BLOCK_K = min(128, triton.next_power_of_2(max_seqlen))

    dtype_size = q.element_size()
    smem_elems = SMEM_BUDGET // dtype_size

    BLOCK_Q = MIN_BLOCK_Q
    BLOCK_K = MIN_BLOCK_K

    while (
        (BLOCK_Q << 1) <= MAX_BLOCK_Q
        and ((BLOCK_Q << 1) * dim + BLOCK_K * dim + (BLOCK_Q << 1) * BLOCK_K <= smem_elems)
    ):
        BLOCK_Q <<= 1

    while (
        (BLOCK_K << 1) <= MAX_BLOCK_K
        and (BLOCK_Q * dim + (BLOCK_K << 1) * dim + BLOCK_Q * (BLOCK_K << 1) <= smem_elems)
    ):
        BLOCK_K <<= 1

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    num_programs = num_groups * math.ceil(max_seqlen / BLOCK_Q)

    while num_programs < num_sms and BLOCK_Q > MIN_BLOCK_Q:
        BLOCK_Q >>= 1
        while (
            (BLOCK_K << 1) <= MAX_BLOCK_K
            and (BLOCK_Q * dim + (BLOCK_K << 1) * dim + BLOCK_Q * (BLOCK_K << 1) <= smem_elems)
        ):
            BLOCK_K <<= 1
        num_programs = num_groups * math.ceil(max_seqlen / BLOCK_Q)

    grid = (num_seqs, num_heads, triton.cdiv(max_seqlen, BLOCK_Q))

    traditional_attention_kernel[grid](
        q, k, v, out,
        cu_seqlens,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        NUM_HEADS=num_heads,
        IS_CAUSAL=is_causal,
        SCALE=scale,
        D=dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
    )
    return out


def varlen_traditional_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Args:
        q: (total_tokens, num_heads, dim)
        k: (total_tokens, num_heads, dim)
        v: (total_tokens, num_heads, dim)
        cu_seqlens: (num_seqs + 1,)
        max_seqlen: int
        scale: float
        is_causal: bool
    Returns:
        out: (total_tokens, num_heads, dim)
    """
    return _traditional_attn(q, k, v, cu_seqlens, max_seqlen, scale, is_causal)


def pack_sequences(
    x: torch.Tensor,
    lengths: torch.Tensor,
):
    """
    Args:
        x: (batch_size, seq_len, dim)
        lengths: (batch_size,)
    Returns:
        packed: (total_tokens, dim)
        cu_seqlens: (batch_size + 1,)
    """
    batch_size, seq_len, dim = x.shape
    lengths = lengths.to(device=x.device, dtype=torch.int32)

    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=x.device),
        lengths.cumsum(0),
    ])

    total_tokens = cu_seqlens[-1].item()
    avg_len = total_tokens // max(1, batch_size)
    block_t = min(128, triton.next_power_of_2(max(1, avg_len)))

    num_token_blocks = triton.cdiv(lengths, block_t)
    pack_offsets = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
    pack_offsets[1:] = num_token_blocks.cumsum(0)
    total_blocks = pack_offsets[-1].item()

    packed = torch.empty(total_tokens, dim, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        total_blocks,
        triton.cdiv(dim, META["BLOCK_D"]),
    )

    pack_sequences_kernel[grid](
        x, packed,
        cu_seqlens, pack_offsets,
        *(x.stride()),
        *(packed.stride()),
        B=batch_size,
        D=dim,
        BLOCK_T=block_t,
    )

    return packed, cu_seqlens


def unpack_sequences(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    """
    Args:
        packed: (total_tokens, dim)
        cu_seqlens: (batch_size + 1,)
    Returns:
        x: (batch_size, max_seq_len, dim)
    """
    batch_size = cu_seqlens.numel() - 1
    total_tokens, dim = packed.shape

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq_len = lengths.max().item()

    x = torch.zeros(
        batch_size,
        max_seq_len,
        dim,
        dtype=packed.dtype,
        device=packed.device,
    )

    avg_len = total_tokens // max(1, batch_size)
    block_t = min(128, triton.next_power_of_2(max(1, avg_len)))

    num_token_blocks = triton.cdiv(lengths, block_t)
    unpack_offsets = torch.zeros(
        batch_size + 1,
        dtype=torch.int32,
        device=packed.device,
    )
    unpack_offsets[1:] = num_token_blocks.cumsum(0)
    total_blocks = unpack_offsets[-1].item()

    grid = lambda META: (
        total_blocks,
        triton.cdiv(dim, META["BLOCK_D"]),
    )

    unpack_sequences_kernel[grid](
        packed,
        x,
        cu_seqlens,
        unpack_offsets,
        *packed.stride(),
        *x.stride(),
        B=batch_size,
        D=dim,
        BLOCK_T=block_t,
    )

    return x