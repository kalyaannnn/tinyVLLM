from __future__ import annotations

import math

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = -x2
    out[..., 1::2] = x1
    return out


def build_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    *,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim))
    freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q: [B,T,Hq,D], k: [B,T,Hk,D], cos/sin: [B,T,D]
    cos_e = cos.unsqueeze(2)
    sin_e = sin.unsqueeze(2)
    q_out = (q * cos_e) + (_rotate_half(q) * sin_e)
    k_out = (k * cos_e) + (_rotate_half(k) * sin_e)
    return q_out, k_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [B,T,Hk,D] -> [B,T,Hk*n_rep,D]
    if n_rep == 1:
        return x
    B, T, Hk, D = x.shape
    return x[:, :, :, None, :].expand(B, T, Hk, n_rep, D).reshape(B, T, Hk * n_rep, D)


@torch.no_grad()
def rope_gqa_attention_ref(
    q: torch.Tensor,  # [B,T,Hq,D]
    k: torch.Tensor,  # [B,T,Hk,D]
    v: torch.Tensor,  # [B,T,Hk,D]
    position_ids: torch.Tensor,  # [B,T]
    *,
    rope_base: float = 10000.0,
    causal: bool = True,
) -> torch.Tensor:
    B, T, Hq, D = q.shape
    Hk = k.shape[2]
    if Hq % Hk != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    cos, sin = build_rope_cos_sin(position_ids, D, base=rope_base)
    q_r, k_r = apply_rope(q, k, cos, sin)
    k_rep = repeat_kv(k_r, Hq // Hk)
    v_rep = repeat_kv(v, Hq // Hk)

    scores = torch.einsum("bthd,bshd->bhts", q_r.float(), k_rep.float()) * (1.0 / math.sqrt(D))
    if causal:
        mask = torch.triu(torch.ones((T, T), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", probs, v_rep.float())
    return out.to(dtype=q.dtype)
