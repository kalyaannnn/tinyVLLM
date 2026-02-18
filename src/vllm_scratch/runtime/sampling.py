from __future__ import annotations
import torch


@torch.no_grad()
def sample_next(
    logits: torch.Tensor,          # [B,V]
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = logits / temperature

    if top_k and top_k > 0:
        vals, idx = torch.topk(x, k=top_k, dim=-1)
        mask = torch.full_like(x, float("-inf"))
        x = mask.scatter(1, idx, vals)

    if top_p < 1.0:
        sorted_x, sorted_idx = torch.sort(x, descending=True, dim=-1)
        probs = torch.softmax(sorted_x, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)
        cut = cdf > top_p
        cut[..., 0] = False
        sorted_x = sorted_x.masked_fill(cut, float("-inf"))
        x = torch.full_like(x, float("-inf")).scatter(1, sorted_idx, sorted_x)

    probs = torch.softmax(x, dim=-1)
    return torch.multinomial(probs, num_samples=1)[:, 0]