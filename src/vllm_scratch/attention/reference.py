from __future__ import annotations

import math
from typing import Optional

import torch


@torch.no_grad()
def paged_attention_decode_ref(
    q : torch.Tensor,
    k_cache : torch.Tensor,
    v_cache : torch.Tensor,
    block_tables : torch.Tensor,
    seqlens : torch.Tensor,
    block_tokens : int,
    attn_mask : Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q.ndim != 3:
        raise ValueError(f"q must be [B,H,D], got {tuple(q.shape)}")
    B, H, D = q.shape
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache shapes must match")
    NB, BT, Hk, Dk = k_cache.shape
    if BT != block_tokens:
        raise ValueError(f"block_tokens mismatch: BT={BT} vs block_tokens={block_tokens}")
    if Hk != H or Dk != D:
        raise ValueError("head/head_dim mismatch between q and cache")


    if block_tables.ndim != 2 or block_tables.shape[0] != B:
        raise ValueError("block_tables must be [B, max_blocks]")
    if seqlens.ndim != 1 or seqlens.shape[0] != B:
        raise ValueError("seqlens must be [B]")


    device = q.device
    out = torch.empty((B, H, D), device = device, dtype = q.dtype)

    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        T = int(seqlens[b].item())
        if T <= 0:
            out[b].zero()
            continue
    
        k_contig = torch.empty((T, H, D), device = device, dtype = q.dtype)
        v_contig = torch.empty((T, H, D), device = device, dtype = q.dtype)

        for t in range(T):
            logical_block = t // block_tokens
            offset = t % block_tokens
            phys = int(block_tables[b, logical_block].item())
            if phys < 0 or phys >= NB:
                raise IndexError(f"Invalid phys block id {phys} for b={b}, t={t}")
            k_contig[t] = k_cache[phys, offset]
            v_contig[t] = v_cache[phys, offset]

        scores = torch.einsum("hd,thd->ht", q[b], k_contig) * scale
        probs = torch.softmax(scores, dim = -1)

        out[b] = torch.einsum("ht,thd->hd", probs, v_contig)


    return out
