from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from vllm_scratch.attention.reference import paged_attention_decode_ref, paged_attention_prefill_ref
from vllm_scratch.attention.triton_decode import paged_attention_decode_triton
from vllm_scratch.attention.triton_prefill_v0 import paged_attention_prefill_triton_v0
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache


@dataclass
class SequenceState:
    bt: BlockTable
    seqlen: int = 0


class ToyEngine:
    """System plumbing test: allocate blocks, write KV, run prefill/decode attention."""
    def __init__(
        self,
        *,
        num_blocks: int,
        block_tokens: int,
        H: int,
        D: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.layer = 0
        self.alloc = BlockAllocator(num_blocks=num_blocks)
        self.block_tokens = block_tokens
        self.H = H
        self.D = D
        self.device = device
        self.dtype = dtype
        self.kv = PagedKVCache(
            num_layers=1,
            num_blocks=num_blocks,
            block_tokens=block_tokens,
            num_heads=H,
            head_dim=D,
            device=device,
            dtype=dtype,
        )

    def new_sequence(self) -> SequenceState:
        return SequenceState(bt=BlockTable(block_tokens=self.block_tokens), seqlen=0)

    def _batch_block_tables(self, seqs: List[SequenceState]) -> torch.Tensor:
        max_blocks = max(len(s.bt.blocks) for s in seqs) if seqs else 0
        bt = torch.full((len(seqs), max_blocks), -1, device=self.device, dtype=torch.int32)
        for i, s in enumerate(seqs):
            for j, phys in enumerate(s.bt.blocks):
                bt[i, j] = int(phys)
        return bt

    def _batch_seqlens(self, seqs: List[SequenceState]) -> torch.Tensor:
        return torch.tensor([s.seqlen for s in seqs], device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def prefill(
        self,
        seqs: List[SequenceState],
        prompt_lens: List[int],
        *,
        use_triton_v0: bool = True,
    ) -> torch.Tensor:
        """Writes KV for prompt tokens (toy random KV), runs prefill attention.
        Returns: [B, Tmax, H, D]
        """
        B = len(seqs)
        Tmax = max(prompt_lens) if B else 0

        # Allocate blocks and write KV
        for s, L in zip(seqs, prompt_lens):
            s.bt.ensure_capacity(L, self.alloc)
            k = torch.randn((L, self.H, self.D), device=self.device, dtype=self.dtype)
            v = torch.randn((L, self.H, self.D), device=self.device, dtype=self.dtype)
            self.kv.write_range(self.layer, s.bt, 0, k, v)
            s.seqlen = L

        # Build Q padded
        q = torch.zeros((B, Tmax, self.H, self.D), device=self.device, dtype=self.dtype)
        seqlens = torch.tensor(prompt_lens, device=self.device, dtype=torch.int32)
        for b, L in enumerate(prompt_lens):
            q[b, :L] = torch.randn((L, self.H, self.D), device=self.device, dtype=self.dtype)

        block_tables = self._batch_block_tables(seqs)
        k_cache = self.kv.k(self.layer)
        v_cache = self.kv.v(self.layer)

        if use_triton_v0:
            return paged_attention_prefill_triton_v0(q, k_cache, v_cache, block_tables, seqlens, self.block_tokens)
        return paged_attention_prefill_ref(q, k_cache, v_cache, block_tables, seqlens, self.block_tokens)

    @torch.no_grad()
    def decode_step(self, seqs: List[SequenceState]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append 1 token KV per seq and compute decode attention output for that token.
        Returns: (q_next [B,H,D], out [B,H,D])
        """
        # Ensure capacity for the new token, then write KV at current position
        for s in seqs:
            s.bt.ensure_capacity(s.seqlen + 1, self.alloc)
            k_t = torch.randn((self.H, self.D), device=self.device, dtype=self.dtype)
            v_t = torch.randn((self.H, self.D), device=self.device, dtype=self.dtype)
            self.kv.write_token(self.layer, s.bt, s.seqlen, k_t, v_t)
            s.seqlen += 1

        q_next = torch.randn((len(seqs), self.H, self.D), device=self.device, dtype=self.dtype)
        block_tables = self._batch_block_tables(seqs)
        seqlens = self._batch_seqlens(seqs)
        k_cache = self.kv.k(self.layer)
        v_cache = self.kv.v(self.layer)

        out = paged_attention_decode_triton(q_next, k_cache, v_cache, block_tables, seqlens, self.block_tokens)
        return q_next, out

    @torch.no_grad()
    def decode_step_ref(self, q_next: torch.Tensor, seqs: List[SequenceState]) -> torch.Tensor:
        block_tables = self._batch_block_tables(seqs)
        seqlens = self._batch_seqlens(seqs)
        k_cache = self.kv.k(self.layer)
        v_cache = self.kv.v(self.layer)
        return paged_attention_decode_ref(q_next, k_cache, v_cache, block_tables, seqlens, self.block_tokens)
