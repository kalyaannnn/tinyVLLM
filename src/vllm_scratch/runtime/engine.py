from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from vllm_scratch.attention.reference import paged_attention_decode_ref
from vllm_scratch.attention.triton_decode import TRITON_AVAILABLE, paged_attention_decode_triton
from vllm_scratch.attention.triton_prefill_v0 import paged_attention_prefill_triton
from vllm_scratch.kv_cache.allocator import BlockAllocator, OutOfBlocks
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
    prefill_budget_reqs: int = 4
    metric_interval_ticks: int = 10


class TinyEngine:
    """Continuous batching runtime with prefill/decode queues and KV preemption."""

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
        self.waiting_to_prefill: List[int] = []
        self.ready_to_decode: List[int] = []
        self.tick = 0
        self._bt_width = 0
        self._bt_rows: Dict[int, torch.Tensor] = {}

    def submit(self, prompt_len: int, max_new_tokens: int, priority: int = 0) -> int:
        rid = self._next_rid
        self._next_rid += 1
        st = RequestState(
            req=Request(
                rid=rid,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                priority=priority,
            ),
            bt=BlockTable(block_tokens=self.cfg.block_tokens),
        )
        self.active[rid] = st
        self.waiting_to_prefill.append(rid)
        return rid

    def _grow_bt_cache_width(self, new_width: int) -> None:
        if new_width <= self._bt_width:
            return
        for rid, row in list(self._bt_rows.items()):
            pad = torch.full(
                (new_width,),
                -1,
                device=self.device,
                dtype=torch.int32,
            )
            pad[: row.numel()] = row
            self._bt_rows[rid] = pad
        self._bt_width = new_width

    def _update_bt_row(self, st: RequestState) -> None:
        width = len(st.bt.blocks)
        if width == 0:
            self._bt_rows.pop(st.req.rid, None)
            return
        self._grow_bt_cache_width(width)
        row = torch.full((self._bt_width,), -1, device=self.device, dtype=torch.int32)
        for i, phys in enumerate(st.bt.blocks):
            row[i] = int(phys)
        self._bt_rows[st.req.rid] = row

    def _pack_block_tables(self, states: List[RequestState]) -> torch.Tensor:
        if not states:
            return torch.empty((0, 0), device=self.device, dtype=torch.int32)
        if self._bt_width == 0:
            return torch.empty((len(states), 0), device=self.device, dtype=torch.int32)
        return torch.stack([self._bt_rows[s.req.rid] for s in states], dim=0)

    def _pack_seqlens(self, states: List[RequestState]) -> torch.Tensor:
        return torch.tensor([s.seqlen for s in states], device=self.device, dtype=torch.int32)

    def _finish_request(self, rid: int) -> None:
        st = self.active.pop(rid)
        st.bt.release(self.alloc)
        self._bt_rows.pop(rid, None)
        if rid in self.waiting_to_prefill:
            self.waiting_to_prefill.remove(rid)
        if rid in self.ready_to_decode:
            self.ready_to_decode.remove(rid)

    def _choose_preemption_victim(self, exclude_rid: int | None = None) -> int | None:
        candidates: List[RequestState] = []
        for rid in self.ready_to_decode:
            if exclude_rid is not None and rid == exclude_rid:
                continue
            st = self.active[rid]
            candidates.append(st)
        if not candidates:
            return None
        candidates.sort(key=lambda s: (s.req.priority, -s.generated, -s.seqlen))
        return candidates[0].req.rid

    def _preempt_one(self, exclude_rid: int | None = None) -> bool:
        victim = self._choose_preemption_victim(exclude_rid=exclude_rid)
        if victim is None:
            return False
        st = self.active[victim]
        st.bt.release(self.alloc)
        st.seqlen = 0
        st.prefilling = True
        st.paused = True
        st.preemptions += 1
        self._bt_rows.pop(victim, None)
        if victim in self.ready_to_decode:
            self.ready_to_decode.remove(victim)
        if victim not in self.waiting_to_prefill:
            self.waiting_to_prefill.insert(0, victim)
        return True

    def _decode_attention(self, q: torch.Tensor, block_tables: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
        if TRITON_AVAILABLE and q.device.type == "cuda":
            return paged_attention_decode_triton(
                q, self.kv.k(0), self.kv.v(0), block_tables, seqlens, self.cfg.block_tokens
            )
        return paged_attention_decode_ref(
            q, self.kv.k(0), self.kv.v(0), block_tables, seqlens, self.cfg.block_tokens
        )

    def _log_metrics(
        self,
        *,
        prefill_ms: float,
        decode_ms: float,
        decode_tokens: int,
        batch_size: int,
        max_seqlen: int,
    ) -> None:
        if self.cfg.metric_interval_ticks <= 0:
            return
        if self.tick % self.cfg.metric_interval_ticks != 0:
            return
        throughput = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        print(
            "[engine] "
            f"tick={self.tick} "
            f"waiting={len(self.waiting_to_prefill)} "
            f"decode_ready={len(self.ready_to_decode)} "
            f"free_blocks={self.alloc.free_count} "
            f"used_blocks={self.alloc.used_count} "
            f"batch={batch_size} "
            f"max_seqlen={max_seqlen} "
            f"prefill_ms={prefill_ms:.2f} "
            f"decode_toks_per_s={throughput:.2f}"
        )

    @torch.no_grad()
    def step(self) -> Tuple[List[int], torch.Tensor]:
        """
        One engine tick:
        1) Prefill a bounded number of waiting requests.
        2) Decode one token for all decode-ready requests.
        """
        self.tick += 1
        if not self.active:
            return [], torch.empty((0, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)

        prefill_start = time.perf_counter()
        prefill_states: List[RequestState] = []
        admit_budget = self.cfg.prefill_budget_reqs

        while self.waiting_to_prefill and len(prefill_states) < admit_budget:
            rid = self.waiting_to_prefill.pop(0)
            st = self.active[rid]
            target_len = st.req.prompt_len + st.generated
            needed_blocks = st.bt.num_logical_blocks_for(target_len)
            missing = needed_blocks - len(st.bt.blocks)
            while missing > self.alloc.free_count:
                if not self._preempt_one(exclude_rid=rid):
                    self.waiting_to_prefill.insert(0, rid)
                    missing = -1
                    break
            if missing == -1:
                break
            try:
                st.bt.ensure_capacity(target_len, self.alloc)
            except OutOfBlocks:
                self.waiting_to_prefill.insert(0, rid)
                break
            self._update_bt_row(st)
            k = torch.randn((target_len, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            v = torch.randn((target_len, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            self.kv.write_range(layer=0, bt=st.bt, start_t=0, k=k, v=v)
            st.seqlen = target_len
            prefill_states.append(st)

        if prefill_states:
            Tmax = max(s.seqlen for s in prefill_states)
            Bp = len(prefill_states)
            q = torch.zeros((Bp, Tmax, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            seqlens = torch.tensor([s.seqlen for s in prefill_states], device=self.device, dtype=torch.int32)
            for i, s in enumerate(prefill_states):
                q[i, : s.seqlen] = torch.randn(
                    (s.seqlen, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype
                )
            block_tables = self._pack_block_tables(prefill_states)
            _ = paged_attention_prefill_triton(
                q, self.kv.k(0), self.kv.v(0), block_tables, seqlens, self.cfg.block_tokens
            )
            for s in prefill_states:
                s.prefilling = False
                s.paused = False
                if s.req.rid not in self.ready_to_decode:
                    self.ready_to_decode.append(s.req.rid)
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
            for s in prefill_states:
                s.prefill_ms += prefill_ms / max(1, len(prefill_states))
        else:
            prefill_ms = 0.0

        decode_start = time.perf_counter()
        decode_states: List[RequestState] = []
        for rid in list(self.ready_to_decode):
            st = self.active[rid]
            try:
                st.bt.ensure_capacity(st.seqlen + 1, self.alloc)
            except OutOfBlocks:
                if not self._preempt_one(exclude_rid=rid):
                    continue
                try:
                    st.bt.ensure_capacity(st.seqlen + 1, self.alloc)
                except OutOfBlocks:
                    continue
            self._update_bt_row(st)
            k_t = torch.randn((self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            v_t = torch.randn((self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            self.kv.write_token(layer=0, bt=st.bt, t=st.seqlen, k_t=k_t, v_t=v_t)
            st.seqlen += 1
            st.generated += 1
            st.decode_tokens += 1
            decode_states.append(st)

        if decode_states:
            B = len(decode_states)
            q_next = torch.randn((B, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            block_tables = self._pack_block_tables(decode_states)
            seqlens = self._pack_seqlens(decode_states)
            decode_out = self._decode_attention(q_next, block_tables, seqlens)
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
            for s in decode_states:
                s.decode_ms += decode_ms / max(1, len(decode_states))
        else:
            decode_out = torch.empty((0, self.cfg.H, self.cfg.D), device=self.device, dtype=self.cfg.dtype)
            decode_ms = 0.0

        done: List[int] = []
        for rid in list(self.ready_to_decode):
            st = self.active[rid]
            if st.generated >= st.req.max_new_tokens:
                done.append(rid)
                self._finish_request(rid)

        max_seqlen = max((s.seqlen for s in self.active.values()), default=0)
        self._log_metrics(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            decode_tokens=len(decode_states),
            batch_size=len(decode_states),
            max_seqlen=max_seqlen,
        )
        return done, decode_out
