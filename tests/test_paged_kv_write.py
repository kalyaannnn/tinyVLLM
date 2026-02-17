import torch

from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache


@torch.no_grad()
def test_paged_kv_write_readback() -> None:
    torch.manual_seed(0)

    device = torch.device("cpu")
    dtype = torch.float32

    L = 2
    H, D = 3, 8
    block_tokens = 4
    seqlen = 11
    num_blocks = 64

    alloc = BlockAllocator(num_blocks=num_blocks)
    bt = BlockTable(block_tokens=block_tokens)
    bt.ensure_capacity(seqlen, alloc)

    kv = PagedKVCache(
        num_layers=L,
        num_blocks=num_blocks,
        block_tokens=block_tokens,
        num_heads=H,
        head_dim=D,
        device=device,
        dtype=dtype,
    )

    # write and verify for each layer
    for layer in range(L):
        ks = torch.randn((seqlen, H, D), device=device, dtype=dtype)
        vs = torch.randn((seqlen, H, D), device=device, dtype=dtype)

        kv.write_range(layer, bt, 0, ks, vs)

        # readback gather
        k_rb = torch.empty_like(ks)
        v_rb = torch.empty_like(vs)
        for t in range(seqlen):
            phys, off = bt.token_slot(t)
            k_rb[t] = kv.k(layer)[phys, off]
            v_rb[t] = kv.v(layer)[phys, off]

        torch.testing.assert_close(k_rb, ks)
        torch.testing.assert_close(v_rb, vs)
