import pytest
import torch

from vllm_scratch.attention.reference import paged_attention_prefill_ref
from vllm_scratch.attention.triton_prefill_v0 import paged_attention_prefill_triton_v0
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_triton_prefill_v0_matches_ref() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    B, H, D = 3, 4, 64
    block_tokens = 16
    seqlens = torch.tensor([33, 7, 49], device=device, dtype=torch.int32)
    T = int(seqlens.max().item())

    # build block tables
    alloc = BlockAllocator(num_blocks=512)
    max_blocks = (T + block_tokens - 1) // block_tokens
    block_tables = torch.full((B, max_blocks), -1, device=device, dtype=torch.int32)

    bts = []
    for b in range(B):
        bt = BlockTable(block_tokens=block_tokens)
        bt.ensure_capacity(int(seqlens[b].item()), alloc)
        bts.append(bt)
        for i, phys in enumerate(bt.blocks):
            block_tables[b, i] = int(phys)

    NB = alloc.num_blocks
    k_cache = torch.zeros((NB, block_tokens, H, D), device=device, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)

    # pack KV for each sequence length
    for b in range(B):
        Tb = int(seqlens[b].item())
        k = torch.randn((Tb, H, D), device=device, dtype=torch.float16)
        v = torch.randn((Tb, H, D), device=device, dtype=torch.float16)
        for t in range(Tb):
            phys, off = bts[b].token_slot(t)
            k_cache[phys, off] = k[t]
            v_cache[phys, off] = v[t]

    # q padded to [B,T,H,D]
    q = torch.zeros((B, T, H, D), device=device, dtype=torch.float16)
    for b in range(B):
        Tb = int(seqlens[b].item())
        q[b, :Tb] = torch.randn((Tb, H, D), device=device, dtype=torch.float16)

    out_ref = paged_attention_prefill_ref(q, k_cache, v_cache, block_tables, seqlens, block_tokens)
    out_v0 = paged_attention_prefill_triton_v0(q, k_cache, v_cache, block_tables, seqlens, block_tokens)

    torch.testing.assert_close(out_v0, out_ref, rtol=2e-2, atol=2e-2)
