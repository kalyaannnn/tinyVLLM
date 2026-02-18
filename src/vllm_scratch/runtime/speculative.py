from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

from vllm_scratch.hf.gpt2_paged import GPT2PagedRunner
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.runtime.sampling import sample_next


@dataclass
class SpeculativeStats:
    proposed: int = 0
    accepted: int = 0
    committed: int = 0


def _pack_block_table_single(device: torch.device, bt: BlockTable) -> torch.Tensor:
    row = torch.full((1, len(bt.blocks)), -1, device=device, dtype=torch.int32)
    for i, phys in enumerate(bt.blocks):
        row[0, i] = int(phys)
    return row


@torch.no_grad()
def draft_propose(
    draft_model: AutoModelForCausalLM,
    draft_input_ids: torch.Tensor,  # [1, T]
    *,
    num_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    props = []
    cur = draft_input_ids
    for _ in range(num_tokens):
        logits = draft_model(cur).logits[:, -1, :]
        nxt = sample_next(logits, temperature=temperature, top_p=top_p)
        props.append(nxt.item())
        cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
    return torch.tensor(props, device=draft_input_ids.device, dtype=torch.long)


@torch.no_grad()
def speculative_verify_commit(
    runner: GPT2PagedRunner,
    alloc: BlockAllocator,
    bt: BlockTable,
    *,
    last_logits: torch.Tensor,  # [V]
    seqlen: int,
    proposed_tokens: torch.Tensor,  # [K]
    temperature: float,
    top_p: float,
    stats: SpeculativeStats,
) -> tuple[list[int], int, torch.Tensor]:
    committed: list[int] = []
    cur_logits = last_logits

    for tok in proposed_tokens.tolist():
        stats.proposed += 1
        target_tok = int(sample_next(cur_logits.view(1, -1), temperature=temperature, top_p=top_p)[0].item())
        if tok == target_tok:
            stats.accepted += 1
            commit_tok = tok
        else:
            commit_tok = target_tok
        committed.append(commit_tok)
        stats.committed += 1

        bt.ensure_capacity(seqlen + 1, alloc)
        seqlen += 1
        block_tables = _pack_block_table_single(runner.device, bt)
        seqlens = torch.tensor([seqlen], device=runner.device, dtype=torch.int32)
        cur_logits = runner.decode_one(
            last_ids=torch.tensor([[commit_tok]], device=runner.device, dtype=torch.long),
            pos_ids=torch.tensor([[seqlen - 1]], device=runner.device, dtype=torch.long),
            bts=[bt],
            block_tables=block_tables,
            seqlens=seqlens,
        )[0]
        if tok != target_tok:
            break

    return committed, seqlen, cur_logits
