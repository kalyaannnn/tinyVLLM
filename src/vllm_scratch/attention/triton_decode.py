from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None

if TRITON_AVAILABLE:

    @triton.jit
    def _paged_attn_decode_kernel(
        Q_ptr, K_ptr, V_ptr, BT_ptr, SL_ptr, O_ptr,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_k0: tl.constexpr, stride_k1: tl.constexpr, stride_k2: tl.constexpr, stride_k3: tl.constexpr,
        stride_v0: tl.constexpr, stride_v1: tl.constexpr, stride_v2: tl.constexpr, stride_v3: tl.constexpr,
        stride_bt0: tl.constexpr, stride_bt1: tl.constexpr,
        stride_o0: tl.constexpr, stride_o1: tl.constexpr, stride_o2: tl.constexpr,
        SCALE: tl.constexpr,
        D: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        seqlen = tl.load(SL_ptr + b).to(tl.int32)

        d = tl.arange(0, D)
        q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + d * stride_qd).to(tl.float32)

        m = tl.full([], -1.0e20, tl.float32)
        l = tl.full([], 0.0, tl.float32)
        o = tl.zeros([D], dtype=tl.float32)

        offs_t = tl.arange(0, BLOCK_TOKENS)

        for blk in tl.static_range(0, MAX_BLOCKS):
            phys = tl.load(BT_ptr + b * stride_bt0 + blk * stride_bt1).to(tl.int32)

            token_pos = blk * BLOCK_TOKENS + offs_t
            tok_mask = token_pos < seqlen
            phys_mask = phys >= 0
            mask = tok_mask & phys_mask

            t2d = offs_t[:, None]
            d2d = d[None, :]
            k_ptrs = (
                K_ptr
                + phys * stride_k0
                + t2d * stride_k1
                + h * stride_k2
                + d2d * stride_k3
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
                V_ptr
                + phys * stride_v0
                + t2d * stride_v1
                + h * stride_v2
                + d2d * stride_v3
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

            o = o * alpha + tl.sum(v * p[:, None], axis=0)

            m = m_new
            l = l_new

        out = tl.where(seqlen > 0, o / l, 0.0)
        tl.store(O_ptr + b * stride_o0 + h * stride_o1 + d * stride_o2, out)
else:
    _paged_attn_decode_kernel = None

def paged_attention_decode_triton(
    q: torch.Tensor,            # [B, H, D], fp16/bf16 recommended
    k_cache: torch.Tensor,      # [NB, BT, H, D]
    v_cache: torch.Tensor,      # [NB, BT, H, D]
    block_tables: torch.Tensor, # [B, MAX_BLOCKS] int32/int64, -1 padded ok
    seqlens: torch.Tensor,      # [B] int32/int64
    block_tokens: int,          # BT
    *,
    num_warps: int = 4,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not installed")
    if q.ndim != 3:
        raise ValueError(f"q must be [B,H,D], got {tuple(q.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache shapes must match")
    if k_cache.ndim != 4:
        raise ValueError("k_cache/v_cache must be [NB,BT,H,D]")
    B, H, D = q.shape
    NB, BT, Hk, Dk = k_cache.shape

    if BT != block_tokens:
        raise ValueError(f"block_tokens mismatch: cache BT={BT} vs block_tokens={block_tokens}")
    if Hk != H or Dk != D:
        raise ValueError("head/head_dim mismatch between q and cache")


    if q.device.type != "cuda":
        raise ValueError("Triton path requires CUDA tensors")

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    seqlens = seqlens.to(torch.int32).contiguous()

    out = torch.empty((B, H, D), device = q.device, dtype = torch.float32)

    sqb, sqh, sqd = q.stride()
    sk0, sk1, sk2, sk3 = k_cache.stride()
    sv0, sv1, sv2, sv3 = v_cache.stride()
    sbt0, sbt1 = block_tables.stride()
    so0, so1, so2 = out.stride()

    grid = (B, H)
    scale = 1.0 / math.sqrt(D)

    _paged_attn_decode_kernel[grid](
        q, k_cache, v_cache, block_tables, seqlens, out,
        stride_qb = sqb, stride_qh = sqh, stride_qd = sqd,
        stride_k0 = sk0, stride_k1 = sk1, stride_k2 = sk2, stride_k3 = sk3,
        stride_v0 = sv0, stride_v1 = sv1, stride_v2 = sv2, stride_v3 = sv3,
        stride_bt0 = sbt0, stride_bt1 = sbt1,
        stride_o0 = so0, stride_o1 = so1, stride_o2 = so2,
        SCALE = scale,
        D = D,
        BLOCK_TOKENS = block_tokens,
        MAX_BLOCKS = block_tables.shape[1],
        num_warps = num_warps,
    )

    return out.to(dtype = q.dtype)
