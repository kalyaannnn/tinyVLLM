import pytest

from vllm_scratch.kv_cache.allocator import BlockAllocator, OutOfBlocks


def test_alloc_free() -> None:
    a = BlockAllocator(num_blocks=3)
    b = a.alloc(2)
    assert len(b) == 2
    assert a.free_count == 1
    a.free(b)
    assert a.free_count == 3


def test_out_of_blocks() -> None:
    a = BlockAllocator(num_blocks=2)
    _ = a.alloc(2)
    with pytest.raises(OutOfBlocks):
        a.alloc(1)
