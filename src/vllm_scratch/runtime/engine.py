from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from vllm_scratch.attention.triton_decode import paged_attention_decode_triton
from vllm_scratch.attention.triton_prefill_v0 import paged_attention_prefill_triton_v0
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache
from vllm_scratch.runtime.request import Request, RequestState


@dataclass
class EngineConfig:
    num_blocks: int = 8192
    block_tokens: int = 16
    H: int = 8
    D: int = 64
    num_layers: int = 1
    dtype: torch.dtype = torch.float16


class TinyEngine:
    """Continuous batching skeleton (toy QKV)."""

    def __init__(self, cfg: EngineConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.alloc = BlockAllocator(num_blocks=cfg.num_blocks)
        self.kv = PagedKVCache(
            num_layers=cfg.num_layers,
            num_blocks=cfg.num_blocks,
            block_tokens=cfg.block_tokens,
            num_heads=cfg.H,
            head_dim=cfg.D,
            device=device,
            dtype=cfg.dtype,
        )
        self._next_rid = 0
        self.active: Dict[int, RequestState] = {}

    def submit(self, prompt_len: int, max_new_tokens: int) -> int:
        rid = self._next_rid
        self._next_rid += 1
        st = RequestState(
            req=Request(rid=rid, prompt_len=prompt_len, max_new_tokens=max_new_tokens),
            bt=BlockTable(block_tokens=self.cfg.block_tokens),
        )
        self.active[rid] = st
        return rid

    def _pack_block_tables(self, states: List[RequestState]) -> torch.Tensor:
        max_blocks = max(len(s.bt.blocks) for s in states) if states else 0
        bt = torch.full((len(states), max_blocks), -1, device=self.device, dtype=torch.int32)
        for i, s in enumerate(states):
            for j, phys in enumerate(s.bt.blocks):
                bt[i, j] = int(phys)
        return bt

    def _pack_seqlens(self, states: List[RequestState]) -> torch.Tensor:
        return torch.tensor([s.seqlen for s in states], device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def step(self) -> Tuple[List[int], torch.Tensor]:
        """
        One engine tick:
          - run prefill for any new requests
          - run one decode step for all active requests
        Returns:
          done_rids: list of requests that finished this tick
          decode_out: [B,H,D] output of decode attention for active batch after step
        """
        if not self.active:
            return [], torch.empty((0, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)

        states = list(self.active.values())

        # --- Prefill stage (toy): write KV for prompt and run prefill attention once ---
        prefill_states = [s for s in states if s.prefilling]
        if prefill_states:
            Tmax = max(s.req.prompt_len for s in prefill_states)
            Bp = len(prefill_states)

            # Allocate blocks + write KV for prompt tokens into layer 0
            for s in prefill_states:
                L = s.req.prompt_len
                s.bt.ensure_capacity(L, self.alloc)
                k = torch.randn((L, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
                v = torch.randn((L, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
                self.kv.write_range(layer=0, bt=s.bt, start_t=0, k=k, v=v)
                s.seqlen = L

            # Q for prompt (toy)
            q = torch.zeros((Bp, Tmax, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            seqlens = torch.tensor([s.req.prompt_len for s in prefill_states], device=self.device, dtype=torch.int32)
            for i, s in enumerate(prefill_states):
                L = s.req.prompt_len
                q[i, :L] = torch.randn((L, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)

            block_tables = self._pack_block_tables(prefill_states)
            _ = paged_attention_prefill_triton_v0(
                q, self.kv.k(0), self.kv.v(0), block_tables, seqlens, self.cfg.block_tokens
            )

            for s in prefill_states:
                s.prefilling = False

        # --- Decode stage: append 1 KV per request, then run decode attention ---
        states = list(self.active.values())
        B = len(states)

        for s in states:
            s.bt.ensure_capacity(s.seqlen + 1, self.alloc)
            k_t = torch.randn((self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            v_t = torch.randn((self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            self.kv.write_token(layer=0, bt=s.bt, t=s.seqlen, k_t=k_t, v_t=v_t)
            s.seqlen += 1
            s.generated += 1

        q_next = torch.randn((B, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
        block_tables = self._pack_block_tables(states)
        seqlens = self._pack_seqlens(states)

        decode_out = paged_attention_decode_triton(
            q_next, self.kv.k(0), self.kv.v(0), block_tables, seqlens, self.cfg.block_tokens
        )

        done = []
        for rid, s in list(self.active.items()):
            if s.generated >= s.req.max_new_tokens:
                done.append(rid)
                del self.active[rid]

        return done, decode_out
