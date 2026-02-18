from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .allocator import BlockAllocator

@dataclass
class BlockTable:
    block_tokens : int
    blocks : List[int] = field(default_factory = list)

    def __post_init__(self) -> None:
        if self.block_tokens <= 0:
            raise ValueError("block_tokens must be > 0")
    
    def num_logical_blocks_for(self, seqlen : int) -> int:
        if seqlen < 0:
            raise ValueError("seqlen must be >= 0")

        return (seqlen + self.block_tokens - 1) // self.block_tokens

    def ensure_capacity(self, seqlen : int, alloc : BlockAllocator) -> None:
        needed = self.num_logical_blocks_for(seqlen)
        missing = needed - len(self.blocks)
        if missing > 0:
            self.blocks.extend(alloc.alloc(missing))

    def token_slot(self, t : int) -> Tuple[int, int]:
        if t < 0:
            raise ValueError("t must be >= 0")
        logical_block = t // self.block_tokens
        offset = t % self.block_tokens
        if logical_block >= len(self.blocks):
            raise IndexError("BlockTable capacity insufficient for token index")
        return self.blocks[logical_block], offset

    def release(self, alloc: BlockAllocator) -> None:
        if self.blocks:
            alloc.free(self.blocks)
            self.blocks.clear()
        
