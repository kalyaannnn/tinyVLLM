import torch

from vllm_scratch.attention.reference import paged_attention_decode_ref
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable


def main() -> None:
    torch.manual_seed(0)

    B, H, D = 2, 2, 8
    block_tokens = 4
    seqlens = torch.tensor([7, 3], dtype=torch.int64)

    # allocate enough physical blocks for both sequences
    total_blocks = sum(((int(t.item()) + block_tokens - 1) // block_tokens) for t in seqlens)
    alloc = BlockAllocator(32)

    # Build block tables
    max_blocks = max(((int(t.item()) + block_tokens - 1) // block_tokens) for t in seqlens)
    block_tables = torch.full((B, max_blocks), -1, dtype=torch.int64)

    bts = []
    for b in range(B):
        bt = BlockTable(block_tokens=block_tokens)
        bt.ensure_capacity(int(seqlens[b].item()), alloc)
        bts.append(bt)
        for i, phys in enumerate(bt.blocks):
            block_tables[b, i] = phys

    # KV caches
    NB = alloc.num_blocks
    k_cache = torch.zeros((NB, block_tokens, H, D), dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)

    # Create contiguous KV and pack into pages
    k_contig = []
    v_contig = []
    for b in range(B):
        T = int(seqlens[b].item())
        k = torch.randn((T, H, D), dtype=torch.float32)
        v = torch.randn((T, H, D), dtype=torch.float32)
        k_contig.append(k)
        v_contig.append(v)

        for t in range(T):
            phys, off = bts[b].token_slot(t)
            k_cache[phys, off] = k[t]
            v_cache[phys, off] = v[t]

    q = torch.randn((B, H, D), dtype=torch.float32)

    out_paged = paged_attention_decode_ref(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seqlens=seqlens,
        block_tokens=block_tokens,
    )

    # Baseline contiguous attention for comparison (per sequence)
    out_base = torch.empty_like(out_paged)
    scale = (D ** -0.5)
    for b in range(B):
        T = int(seqlens[b].item())
        scores = torch.einsum("hd,thd->ht", q[b], k_contig[b]) * scale
        probs = torch.softmax(scores, dim=-1)
        out_base[b] = torch.einsum("ht,thd->hd", probs, v_contig[b])

    torch.testing.assert_close(out_paged, out_base, rtol=1e-5, atol=1e-6)
    print("CPU smoke OK (paged ref matches contiguous baseline).")


if __name__ == "__main__":
    main()
