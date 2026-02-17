from __future__ import annotations

import torch

from vllm_scratch.attention.triton_decode import paged_attention_decode_triton


@torch.no_grad()
def paged_attention_prefill_triton_v0(
    q: torch.Tensor,                  # [B, T, H, D]
    k_cache: torch.Tensor,            # [NB, BT, H, D]
    v_cache: torch.Tensor,            # [NB, BT, H, D]
    block_tables: torch.Tensor,       # [B, max_blocks]
    seqlens: torch.Tensor,            # [B]
    block_tokens: int,                # BT
) -> torch.Tensor:
    """
    Prefill via repeated decode:
      out[:, t] = decode(q[:, t], attend to keys [0..t]).

    This is a correctness stepping stone.
    Performance comes later with a fused prefill kernel.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,D], got {tuple(q.shape)}")
    B, T, H, D = q.shape
    device = q.device

    seqlens_i32 = seqlens.to(torch.int32)
    out = torch.zeros((B, T, H, D), device=device, dtype=q.dtype)

    for t in range(T):
        # only sequences where t < seqlen are valid queries
        active = (seqlens_i32 > t)
        if not bool(active.any().item()):
            break

        q_t = q[:, t].contiguous()  # [B,H,D]
        seqlens_step = torch.minimum(seqlens_i32, torch.full_like(seqlens_i32, t + 1))
        out_t = paged_attention_decode_triton(
            q_t, k_cache, v_cache, block_tables, seqlens_step, block_tokens
        )  # [B,H,D]
        out_t = torch.where(active[:, None, None], out_t, torch.zeros_like(out_t))
        out[:, t] = out_t

    return out
