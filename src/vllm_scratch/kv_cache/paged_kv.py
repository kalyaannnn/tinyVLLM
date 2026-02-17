from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from vllm_scratch.kv_cache.block_table import BlockTable


@dataclass
class PagedKVCache:
    """Paged KV storage for all layers.
    Block IDs are shared across layers; each layer has its own K/V tensors indexed by block id.
    Shapes per layer:
      K[l]: [num_blocks, block_tokens, H, D]
      V[l]: [num_blocks, block_tokens, H, D]
    """
    num_layers: int
    num_blocks: int
    block_tokens: int
    num_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self) -> None:
        self.k_layers: List[torch.Tensor] = []
        self.v_layers: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            k = torch.empty(
                (self.num_blocks, self.block_tokens, self.num_heads, self.head_dim),
                device=self.device,
                dtype=self.dtype,
            )
            v = torch.empty_like(k)
            self.k_layers.append(k)
            self.v_layers.append(v)

    def write_token(
        self,
        layer: int,
        bt: BlockTable,
        t: int,
        k_t: torch.Tensor,  # [H,D]
        v_t: torch.Tensor,  # [H,D]
    ) -> None:
        phys, off = bt.token_slot(t)
        self.k_layers[layer][phys, off].copy_(k_t)
        self.v_layers[layer][phys, off].copy_(v_t)

    def write_range(
        self,
        layer: int,
        bt: BlockTable,
        start_t: int,
        k: torch.Tensor,  # [T,H,D]
        v: torch.Tensor,  # [T,H,D]
    ) -> None:
        T = k.shape[0]
        for i in range(T):
            self.write_token(layer, bt, start_t + i, k[i], v[i])

    def k(self, layer: int) -> torch.Tensor:
        return self.k_layers[layer]

    def v(self, layer: int) -> torch.Tensor:
        return self.v_layers[layer]
