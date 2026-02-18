from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_scratch.hf.gpt2_paged import GPT2PagedRunner
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache
from vllm_scratch.runtime.sampling import sample_next


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _pack_tables(device: torch.device, bts: list[BlockTable], block_tokens: int, seqlens: torch.Tensor) -> torch.Tensor:
    max_seqlen = int(seqlens.max().item())
    max_blocks = (max_seqlen + block_tokens - 1) // block_tokens
    out = torch.full((len(bts), max_blocks), -1, device=device, dtype=torch.int32)
    for i, bt in enumerate(bts):
        for j, phys in enumerate(bt.blocks):
            out[i, j] = int(phys)
    return out


@torch.no_grad()
def run_case(
    *,
    model,
    tok,
    prompts: list[str],
    max_new: int,
    block_tokens: int,
    num_blocks: int,
    temperature: float,
    top_p: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    seqlens = attn_mask.sum(dim=1).to(torch.int32)
    B = input_ids.shape[0]

    alloc = BlockAllocator(num_blocks=num_blocks)
    bts = [BlockTable(block_tokens=block_tokens) for _ in range(B)]
    for b in range(B):
        bts[b].ensure_capacity(int(seqlens[b].item()), alloc)

    n_layer = int(model.config.n_layer)
    n_embd = int(model.config.n_embd)
    n_head = (
        model.transformer.h[0].attn.n_head
        if hasattr(model.transformer.h[0].attn, "n_head")
        else model.transformer.h[0].attn.num_heads
    )
    kv = PagedKVCache(
        num_layers=n_layer,
        num_blocks=num_blocks,
        block_tokens=block_tokens,
        num_heads=n_head,
        head_dim=n_embd // n_head,
        device=device,
        dtype=dtype,
    )
    runner = GPT2PagedRunner(model=model, kv=kv, block_tokens=block_tokens)

    pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0).expand(B, -1)
    bt = _pack_tables(device, bts, block_tokens, seqlens)
    t0 = time.perf_counter()
    logits = runner.prefill(input_ids, pos, bts, bt, seqlens)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    last_logits = logits[torch.arange(B, device=device), seqlens - 1]
    cur = sample_next(last_logits, temperature=temperature, top_p=top_p)
    active = torch.ones((B,), device=device, dtype=torch.bool)
    gen = torch.zeros((B,), device=device, dtype=torch.int32)

    decode_tokens = 0
    t1 = time.perf_counter()
    while bool(active.any().item()):
        ids = torch.nonzero(active, as_tuple=False).view(-1)
        decode_tokens += int(ids.numel())
        for b in ids.tolist():
            bts[b].ensure_capacity(int(seqlens[b].item()) + 1, alloc)
            gen[b] += 1
            if int(cur[b].item()) == tok.eos_token_id or gen[b] >= max_new:
                active[b] = False
                bts[b].release(alloc)
        if not bool(active.any().item()):
            break
        ids = torch.nonzero(active, as_tuple=False).view(-1)
        seqlens[ids] += 1
        bt_act = [bts[i] for i in ids.tolist()]
        logits_next = runner.decode_one(
            last_ids=cur[ids].view(-1, 1),
            pos_ids=(seqlens[ids] - 1).view(-1, 1),
            bts=bt_act,
            block_tables=_pack_tables(device, bt_act, block_tokens, seqlens[ids]),
            seqlens=seqlens[ids],
        )
        cur[ids] = sample_next(logits_next, temperature=temperature, top_p=top_p)
    decode_ms = (time.perf_counter() - t1) * 1000.0
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "decode_toks_s": decode_tokens / (decode_ms / 1000.0),
        "total_ms": prefill_ms + decode_ms,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="distilgpt2,gpt2")
    ap.add_argument("--batch-sizes", type=str, default="1,2,4")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=8192)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)

    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a distant future, humanity discovers",
        "The capital of California is",
        "A concise summary of attention mechanisms:",
    ]

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n## model={model_name}")
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()
        for bs in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
            prompts = [base_prompts[i % len(base_prompts)] for i in range(bs)]
            torch.cuda.reset_peak_memory_stats(device)
            m = run_case(
                model=model,
                tok=tok,
                prompts=prompts,
                max_new=args.max_new,
                block_tokens=args.block_tokens,
                num_blocks=args.num_blocks,
                temperature=0.9,
                top_p=0.9,
                dtype=dtype,
                device=device,
            )
            print(
                f"batch={bs} prefill_ms={m['prefill_ms']:.2f} decode_ms={m['decode_ms']:.2f} "
                f"decode_toks_s={m['decode_toks_s']:.2f} peak_mem_gb={m['peak_mem_gb']:.3f} total_ms={m['total_ms']:.2f}"
            )


if __name__ == "__main__":
    main()
