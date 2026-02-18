from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

from vllm_scratch.attention.triton_decode import paged_attention_decode_triton
from vllm_scratch.attention.triton_prefill_v0 import paged_attention_prefill_triton
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache


def _get_heads(attn) -> int:
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    if hasattr(attn, "n_head"):
        return int(attn.n_head)
    raise AttributeError("Cannot find num_heads/n_head on attention module")


def _shape_qkv(x: torch.Tensor, n_head: int) -> torch.Tensor:
    # [B,T,C] -> [B,T,H,D]
    B, T, C = x.shape
    assert C % n_head == 0
    D = C // n_head
    return x.view(B, T, n_head, D)


@dataclass
class GPT2PagedRunner:
    model: AutoModelForCausalLM
    kv: PagedKVCache
    block_tokens: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def n_layer(self) -> int:
        return int(self.model.config.n_layer)

    @property
    def n_embd(self) -> int:
        return int(self.model.config.n_embd)

    @property
    def n_head(self) -> int:
        return _get_heads(self.model.transformer.h[0].attn)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,          # [B,T]
        pos_ids: torch.Tensor,            # [B,T]
        bts: list[BlockTable],            # length B
        block_tables: torch.Tensor,       # [B,max_blocks]
        seqlens: torch.Tensor,            # [B]
    ) -> torch.Tensor:
        """
        Writes KV for ALL layers for the prompt and returns logits for all positions: [B,T,V].
        """
        B, T = input_ids.shape
        dev = self.device
        dtype = self.dtype

        # embeddings
        hs = self.model.transformer.wte(input_ids) + self.model.transformer.wpe(pos_ids)
        hs = self.model.transformer.drop(hs)

        for l in range(self.n_layer):
            block = self.model.transformer.h[l]
            attn = block.attn

            # attention
            residual = hs
            x = block.ln_1(hs)
            qkv = attn.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            q = _shape_qkv(q, self.n_head)  # [B,T,H,D]
            k = _shape_qkv(k, self.n_head)
            v = _shape_qkv(v, self.n_head)

            # write KV into paged cache (per sequence)
            for b in range(B):
                Tb = int(seqlens[b].item())
                self.kv.write_range(layer=l, bt=bts[b], start_t=0, k=k[b, :Tb].contiguous(), v=v[b, :Tb].contiguous())

            attn_out = paged_attention_prefill_triton(
                q=q,
                k_cache=self.kv.k(l),
                v_cache=self.kv.v(l),
                block_tables=block_tables,
                seqlens=seqlens,
                block_tokens=self.block_tokens,
            )  # [B,T,H,D]

            attn_out_proj = attn.c_proj(attn_out.reshape(B, T, self.n_embd))
            hs = residual + attn_out_proj

            # mlp
            residual = hs
            x = block.ln_2(hs)
            hs = residual + block.mlp(x)

        hs = self.model.transformer.ln_f(hs)
        logits = self.model.lm_head(hs)  # [B,T,V]
        return logits

    @torch.no_grad()
    def decode_one(
        self,
        last_ids: torch.Tensor,           # [B,1]
        pos_ids: torch.Tensor,            # [B,1]
        bts: list[BlockTable],
        block_tables: torch.Tensor,       # [B,max_blocks]
        seqlens: torch.Tensor,            # [B]
    ) -> torch.Tensor:
        """
        Appends one KV per layer (at current position) and returns logits for the new token: [B,V].
        """
        B = last_ids.shape[0]
        dev = self.device

        # embeddings for the new position
        hs = self.model.transformer.wte(last_ids) + self.model.transformer.wpe(pos_ids)
        hs = self.model.transformer.drop(hs)  # [B,1,C]

        for l in range(self.n_layer):
            block = self.model.transformer.h[l]
            attn = block.attn

            residual = hs
            x = block.ln_1(hs)                # [B,1,C]
            qkv = attn.c_attn(x)              # [B,1,3C]
            q, k, v = qkv.split(self.n_embd, dim=2)

            q = _shape_qkv(q, self.n_head)    # [B,1,H,D]
            k = _shape_qkv(k, self.n_head)
            v = _shape_qkv(v, self.n_head)

            # write new KV at index seqlens[b]-1 (caller already incremented seqlens)
            for b in range(B):
                t = int(seqlens[b].item()) - 1
                self.kv.write_token(layer=l, bt=bts[b], t=t, k_t=k[b, 0].contiguous(), v_t=v[b, 0].contiguous())

            attn_out = paged_attention_decode_triton(
                q=q[:, 0],                     # [B,H,D]
                k_cache=self.kv.k(l),
                v_cache=self.kv.v(l),
                block_tables=block_tables,
                seqlens=seqlens,
                block_tokens=self.block_tokens,
            )  # [B,H,D]

            attn_out_proj = attn.c_proj(attn_out.reshape(B, self.n_embd)).unsqueeze(1)  # [B,1,C]
            hs = residual + attn_out_proj

            residual = hs
            x = block.ln_2(hs)
            hs = residual + block.mlp(x)

        hs = self.model.transformer.ln_f(hs)           # [B,1,C]
        logits = self.model.lm_head(hs)[:, 0]          # [B,V]
        return logits
