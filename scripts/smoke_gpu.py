import torch

from vllm_scratch.attention.reference import paged_attention_decode_ref


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available (expected on Mac). Exiting OK.")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)

    B, H, D = 2, 4, 32
    block_tokens = 16
    seqlens = torch.tensor([128, 96], device=device, dtype=torch.int64)

    max_blocks = int(((int(seqlens.max().item()) + block_tokens - 1) // block_tokens))
    block_tables = torch.arange(0, B * max_blocks, device=device, dtype=torch.int64).view(B, max_blocks)

    NB = int(block_tables.max().item() + 1)
    k_cache = torch.randn((NB, block_tokens, H, D), device=device, dtype=torch.float16).to(torch.float32)
    v_cache = torch.randn((NB, block_tokens, H, D), device=device, dtype=torch.float16).to(torch.float32)
    q = torch.randn((B, H, D), device=device, dtype=torch.float16).to(torch.float32)

    out = paged_attention_decode_ref(q, k_cache, v_cache, block_tables, seqlens, block_tokens)
    assert torch.isfinite(out).all()
    print("GPU smoke OK (reference ran, finite outputs).")


if __name__ == "__main__":
    main()
