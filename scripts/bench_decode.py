from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

from vllm_scratch.attention.reference import paged_attention_decode_ref
from vllm_scratch.attention.triton_decode import paged_attention_decode_triton
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable


@dataclass
class BenchResult:
    name: str
    seqlen: int
    ms: float
    tok_s: float
    kv_tok_s: float


def _parse_seqlens(s: str) -> List[int]:
    # e.g. "32,128,512,1024"
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Empty seqlens list")
    return out


@torch.no_grad()
def _build_paged_kv(
    device: torch.device,
    B: int,
    H: int,
    D: int,
    block_tokens: int,
    max_seqlen: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build:
      - k_cache, v_cache: [NB, BT, H, D]
      - block_tables:     [B, max_blocks]  (no -1 padding needed here)
    with each sequence getting its own blocks.
    """
    max_blocks = (max_seqlen + block_tokens - 1) // block_tokens
    num_blocks = B * max_blocks + 32  # slack

    alloc = BlockAllocator(num_blocks=num_blocks)
    block_tables = torch.full((B, max_blocks), -1, device=device, dtype=torch.int32)

    # Allocate per-sequence block tables
    bts: List[BlockTable] = []
    for b in range(B):
        bt = BlockTable(block_tokens=block_tokens)
        bt.ensure_capacity(max_seqlen, alloc)
        bts.append(bt)
        for i, phys in enumerate(bt.blocks):
            block_tables[b, i] = int(phys)

    NB = alloc.num_blocks
    k_cache = torch.empty((NB, block_tokens, H, D), device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    # Fill the KV cache for tokens [0, max_seqlen) for each sequence
    # (values only matter for perf; correctness checked separately)
    for b in range(B):
        for t in range(max_seqlen):
            phys, off = bts[b].token_slot(t)
            # deterministic-ish fill without huge rand overhead
            k_cache[phys, off].normal_(mean=0.0, std=1.0)
            v_cache[phys, off].normal_(mean=0.0, std=1.0)

    return k_cache, v_cache, block_tables


@torch.no_grad()
def _time_cuda(fn, warmup: int, iters: int) -> float:
    # Returns average ms/iter
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


def _fmt(results: List[BenchResult]) -> str:
    # Simple aligned text table
    cols = ["impl", "seqlen", "ms/step", "tok/s", "kv_tok/s"]
    rows = []
    for r in results:
        rows.append(
            [
                r.name,
                str(r.seqlen),
                f"{r.ms:.3f}",
                f"{r.tok_s:,.0f}",
                f"{r.kv_tok_s:,.0f}",
            ]
        )
    widths = [max(len(c), max(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "  ".join("-" * widths[i] for i in range(len(cols)))
    body = "\n".join("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))) for row in rows)
    return f"{line}\n{sep}\n{body}"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--block_tokens", type=int, default=16)
    parser.add_argument("--seqlens", type=str, default="32,128,512,1024,2048,4096")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--ref_max_seqlen", type=int, default=256)  # ref is very slow
    parser.add_argument("--check", action="store_true", help="Run one correctness check triton vs ref on small seqlen")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Run this on Colab/A100.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    seqlens = _parse_seqlens(args.seqlens)
    max_seqlen = max(seqlens)

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("B,H,D:", args.B, args.H, args.D, "block_tokens:", args.block_tokens, "dtype:", args.dtype)
    print("seqlens:", seqlens)
    print()

    # Build max-sized paged KV + block tables once
    k_cache, v_cache, block_tables = _build_paged_kv(
        device=device,
        B=args.B,
        H=args.H,
        D=args.D,
        block_tokens=args.block_tokens,
        max_seqlen=max_seqlen,
        dtype=dtype,
    )

    # One fixed query tensor reused for timing
    q = torch.randn((args.B, args.H, args.D), device=device, dtype=dtype)

    results: List[BenchResult] = []

    # Optional correctness check on a small seqlen
    if args.check:
        small_T = min(seqlens)
        seqlens_t = torch.full((args.B,), small_T, device=device, dtype=torch.int32)
        out_ref = paged_attention_decode_ref(q, k_cache, v_cache, block_tables, seqlens_t, args.block_tokens)
        out_tri = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, seqlens_t, args.block_tokens)
        torch.testing.assert_close(out_tri, out_ref, rtol=2e-2, atol=2e-2)
        print(f"check OK (triton matches ref) at seqlen={small_T}")
        print()

    # Benchmark Triton across all seqlens
    for T in seqlens:
        seqlens_t = torch.full((args.B,), T, device=device, dtype=torch.int32)

        def run_triton():
            _ = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, seqlens_t, args.block_tokens)

        ms = _time_cuda(run_triton, warmup=args.warmup, iters=args.iters)
        tok_s = args.B * 1000.0 / ms
        kv_tok_s = (args.B * T) * 1000.0 / ms
        results.append(BenchResult("triton", T, ms, tok_s, kv_tok_s))

    # Benchmark reference only for small seqlens (it’s Python-loop heavy)
    for T in seqlens:
        if T > args.ref_max_seqlen:
            continue
        seqlens_t = torch.full((args.B,), T, device=device, dtype=torch.int32)

        def run_ref():
            _ = paged_attention_decode_ref(q, k_cache, v_cache, block_tables, seqlens_t, args.block_tokens)

        # fewer iters for ref to keep runtime reasonable
        ms = _time_cuda(run_ref, warmup=max(5, args.warmup // 4), iters=max(10, args.iters // 10))
        tok_s = args.B * 1000.0 / ms
        kv_tok_s = (args.B * T) * 1000.0 / ms
        results.append(BenchResult("ref", T, ms, tok_s, kv_tok_s))

    # Print grouped (triton first, then ref)
    results_sorted = sorted(results, key=lambda r: (0 if r.name == "triton" else 1, r.seqlen))
    print(_fmt(results_sorted))
    print()
    print("Notes:")
    print("- tok/s = B tokens per decode step.")
    print("- kv_tok/s = B*T key/value positions processed per step (useful for scaling with seqlen).")
    print("- ref is expected to be much slower; it exists as a correctness oracle.")


if __name__ == "__main__":
    main()
