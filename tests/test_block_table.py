from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable


def test_block_table_mapping() -> None:
    alloc = BlockAllocator(num_blocks=10)
    bt = BlockTable(block_tokens=4)

    bt.ensure_capacity(9, alloc)  # needs 3 blocks
    assert len(bt.blocks) == 3

    b0, o0 = bt.token_slot(0)
    assert o0 == 0

    b4, o4 = bt.token_slot(4)
    assert o4 == 0
    assert b4 != b0


def test_block_table_release_returns_capacity() -> None:
    alloc = BlockAllocator(num_blocks=6)
    bt = BlockTable(block_tokens=4)

    bt.ensure_capacity(13, alloc)  # 4 blocks
    assert len(bt.blocks) == 4
    assert alloc.free_count == 2

    bt.release(alloc)
    assert bt.blocks == []
    assert alloc.free_count == 6
