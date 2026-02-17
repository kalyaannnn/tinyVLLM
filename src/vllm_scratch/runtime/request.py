from __future__ import annotations
from dataclasses import dataclass

from vllm_scratch.kv_cache.block_table import BlockTable


@dataclass
class Request:
    rid: int
    prompt_len: int
    max_new_tokens: int


@dataclass
class RequestState:
    req: Request
    bt: BlockTable
    seqlen: int = 0          # total tokens currently in cache
    generated: int = 0       # decode tokens generated so far
    prefilling: bool = True  # still needs prefill
