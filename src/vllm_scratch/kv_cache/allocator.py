from __future__ import annotations

from dataclasses import dataclass
from typing import List


class OutOfBlocks(RuntimeError):
    pass


@dataclass
class BlockAllocator:
    num_blocks: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        self._free: List[int] = list(range(self.num_blocks - 1, -1, -1))

    @property
    def free_count(self) -> int:
        return len(self._free)

    def alloc(self, n: int = 1) -> List[int]:
        if n <= 0:
            raise ValueError("n must be > 0")
        if len(self._free) < n:
            raise OutOfBlocks(f"Requested {n} blocks, only {len(self._free)} free.")
        return [self._free.pop() for _ in range(n)]

    def free(self, block_ids: List[int]) -> None:
        for b in block_ids:
            if not (0 <= b < self.num_blocks):
                raise ValueError(f"Invalid block id {b}")
            self._free.append(b)
