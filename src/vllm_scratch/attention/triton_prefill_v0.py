from __future__ import annotations

import math

import torch

from vllm_scratch.attention.reference import paged_attention_prefill_ref

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None


if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attn_prefill_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        BT_ptr,
        SL_ptr,
        O_ptr,
        stride_qb: tl.constexpr,
        stride_qt: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_k0: tl.constexpr,
        stride_k1: tl.constexpr,
        stride_k2: tl.constexpr,
        stride_k3: tl.constexpr,
        stride_v0: tl.constexpr,
        stride_v1: tl.constexpr,
        stride_v2: tl.constexpr,
        stride_v3: tl.constexpr,
        stride_bt0: tl.constexpr,
        stride_bt1: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_ot: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_od: tl.constexpr,
        SCALE: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        t = tl.program_id(2)

        seqlen = tl.load(SL_ptr + b).to(tl.int32)
        d = tl.arange(0, D)

        valid_q = t < seqlen
        q = tl.load(
            Q_ptr + b * stride_qb + t * stride_qt + h * stride_qh + d * stride_qd,
            mask=valid_q,
            other=0.0,
        ).to(tl.float32)

        m = tl.full([], -1.0e20, tl.float32)
        l = tl.full([], 0.0, tl.float32)
        o = tl.zeros([D], dtype=tl.float32)

        offs_t = tl.arange(0, BLOCK_TOKENS)

        for blk in tl.static_range(0, MAX_BLOCKS):
            phys = tl.load(BT_ptr + b * stride_bt0 + blk * stride_bt1).to(tl.int32)
            token_pos = blk * BLOCK_TOKENS + offs_t
            causal = token_pos <= t
            in_seq = token_pos < seqlen
            valid_phys = phys >= 0
            mask = causal & in_seq & valid_phys & valid_q

            t2d = offs_t[:, None]
            d2d = d[None, :]
            k_ptrs = (
                K_ptr + phys * stride_k0 + t2d * stride_k1 + h * stride_k2 + d2d * stride_k3
            )
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * SCALE
            scores = tl.where(mask, scores, -1.0e20)

            blk_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m, blk_max)
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask, p, 0.0)

            l_new = l * alpha + tl.sum(p, axis=0)
            v_ptrs = (
                V_ptr + phys * stride_v0 + t2d * stride_v1 + h * stride_v2 + d2d * stride_v3
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            o = o * alpha + tl.sum(v * p[:, None], axis=0)

            m = m_new
            l = l_new

        out = tl.where(valid_q & (l > 0), o / l, 0.0)
        tl.store(O_ptr + b * stride_ob + t * stride_ot + h * stride_oh + d * stride_od, out)
else:
    _paged_attn_prefill_kernel = None


@torch.no_grad()
def paged_attention_prefill_triton(
    q: torch.Tensor,  # [B, T, H, D]
    k_cache: torch.Tensor,  # [NB, BT, H, D]
    v_cache: torch.Tensor,  # [NB, BT, H, D]
    block_tables: torch.Tensor,  # [B, max_blocks]
    seqlens: torch.Tensor,  # [B]
    block_tokens: int,  # BT
    *,
    num_warps: int = 4,
) -> torch.Tensor:
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,D], got {tuple(q.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache shapes must match")
    if k_cache.ndim != 4:
        raise ValueError("k_cache/v_cache must be [NB,BT,H,D]")
    B, T, H, D = q.shape
    NB, BT, Hk, Dk = k_cache.shape
    if BT != block_tokens:
        raise ValueError(f"block_tokens mismatch: cache BT={BT} vs block_tokens={block_tokens}")
    if Hk != H or Dk != D:
        raise ValueError("head/head_dim mismatch between q and cache")
    if block_tables.ndim != 2 or block_tables.shape[0] != B:
        raise ValueError("block_tables must be [B, max_blocks]")
    if seqlens.ndim != 1 or seqlens.shape[0] != B:
        raise ValueError("seqlens must be [B]")

    if not TRITON_AVAILABLE or q.device.type != "cuda":
        return paged_attention_prefill_ref(q, k_cache, v_cache, block_tables, seqlens, block_tokens)

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    seqlens = seqlens.to(torch.int32).contiguous()
    out = torch.empty_like(q, device=q.device, dtype=torch.float32)

    sqb, sqt, sqh, sqd = q.stride()
    sk0, sk1, sk2, sk3 = k_cache.stride()
    sv0, sv1, sv2, sv3 = v_cache.stride()
    sbt0, sbt1 = block_tables.stride()
    sob, sot, soh, sod = out.stride()

    grid = (B, H, T)
    _paged_attn_prefill_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_tables,
        seqlens,
        out,
        stride_qb=sqb,
        stride_qt=sqt,
        stride_qh=sqh,
        stride_qd=sqd,
        stride_k0=sk0,
        stride_k1=sk1,
        stride_k2=sk2,
        stride_k3=sk3,
        stride_v0=sv0,
        stride_v1=sv1,
        stride_v2=sv2,
        stride_v3=sv3,
        stride_bt0=sbt0,
        stride_bt1=sbt1,
        stride_ob=sob,
        stride_ot=sot,
        stride_oh=soh,
        stride_od=sod,
        SCALE=1.0 / math.sqrt(D),
        T=T,
        D=D,
        BLOCK_TOKENS=block_tokens,
        MAX_BLOCKS=block_tables.shape[1],
        num_warps=num_warps,
    )
    return out.to(dtype=q.dtype)


@torch.no_grad()
def paged_attention_prefill_triton_v0(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seqlens: torch.Tensor,
    block_tokens: int,
) -> torch.Tensor:
    # Backward-compatible alias.
    return paged_attention_prefill_triton(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seqlens=seqlens,
        block_tokens=block_tokens,
    )
