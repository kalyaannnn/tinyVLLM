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
from vllm_scratch.runtime.speculative import SpeculativeStats, draft_propose, speculative_verify_commit


def pack_block_tables(
    device: torch.device,
    bts: list[BlockTable],
    block_tokens: int,
    seqlens: torch.Tensor,
) -> torch.Tensor:
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    max_blocks = (max_seqlen + block_tokens - 1) // block_tokens if max_seqlen > 0 else 0
    bt = torch.full((len(bts), max_blocks), -1, device=device, dtype=torch.int32)
    for i, st in enumerate(bts):
        for j, phys in enumerate(st.blocks):
            bt[i, j] = int(phys)
    return bt


def _load_dtype(dtype: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--draft-model", type=str, default="")
    ap.add_argument("--prompts", type=str, default="The quick brown fox||In a distant future,||NYU is located in")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--spec-draft-tokens", type=int, default=4)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=8192)
    ap.add_argument("--metrics-every", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    dtype = _load_dtype(args.dtype)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    draft_model = None
    if args.draft_model:
        draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype=dtype).to(device)
        draft_model.eval()

    prompts = [p.strip() for p in args.prompts.split("||") if p.strip()]
    if not prompts:
        raise ValueError("Need at least one prompt")
    B = len(prompts)

    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    seqlens = attn_mask.sum(dim=1).to(torch.int32)

    alloc = BlockAllocator(num_blocks=args.num_blocks)
    bts = [BlockTable(block_tokens=args.block_tokens) for _ in range(B)]
    for b in range(B):
        bts[b].ensure_capacity(int(seqlens[b].item()), alloc)

    n_layer = int(model.config.n_layer)
    n_embd = int(model.config.n_embd)
    n_head = (
        model.transformer.h[0].attn.n_head
        if hasattr(model.transformer.h[0].attn, "n_head")
        else model.transformer.h[0].attn.num_heads
    )
    head_dim = n_embd // n_head
    kv = PagedKVCache(
        num_layers=n_layer,
        num_blocks=args.num_blocks,
        block_tokens=args.block_tokens,
        num_heads=n_head,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    runner = GPT2PagedRunner(model=model, kv=kv, block_tokens=args.block_tokens)

    pos = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0).expand(B, -1)
    bt_mat = pack_block_tables(device, bts, args.block_tokens, seqlens)
    t_prefill0 = time.perf_counter()
    logits = runner.prefill(input_ids, pos, bts, bt_mat, seqlens)
    prefill_ms = (time.perf_counter() - t_prefill0) * 1000.0

    last_logits = logits[torch.arange(B, device=device), seqlens - 1]
    cur_tokens = sample_next(last_logits, temperature=args.temperature, top_p=args.top_p)
    generated: list[list[int]] = [[] for _ in range(B)]
    prompt_ids_unpadded = [input_ids[b, : int(seqlens[b].item())].clone() for b in range(B)]
    active = torch.ones((B,), device=device, dtype=torch.bool)
    gen_counts = torch.zeros((B,), device=device, dtype=torch.int32)
    spec_stats = SpeculativeStats()

    decode_tokens_total = 0
    decode_ms_total = 0.0
    steps = 0
    while bool(active.any().item()):
        steps += 1
        t0 = time.perf_counter()
        if draft_model is not None and args.spec_draft_tokens > 1:
            for b in range(B):
                if not active[b]:
                    continue
                proposed = draft_propose(
                    draft_model,
                    prompt_ids_unpadded[b].view(1, -1),
                    num_tokens=args.spec_draft_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                committed, new_seqlen, new_logits = speculative_verify_commit(
                    runner=runner,
                    alloc=alloc,
                    bt=bts[b],
                    last_logits=last_logits[b],
                    seqlen=int(seqlens[b].item()),
                    proposed_tokens=proposed,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stats=spec_stats,
                )
                if committed:
                    toks = torch.tensor(committed, device=device, dtype=torch.long)
                    prompt_ids_unpadded[b] = torch.cat([prompt_ids_unpadded[b], toks], dim=0)
                    generated[b].extend(committed)
                    gen_counts[b] += len(committed)
                    decode_tokens_total += len(committed)
                    seqlens[b] = int(new_seqlen)
                    last_logits[b] = new_logits
                    if tok.eos_token_id in committed or gen_counts[b] >= args.max_new:
                        active[b] = False
                        bts[b].release(alloc)
        else:
            active_idx = torch.nonzero(active, as_tuple=False).view(-1)
            decode_tokens_total += int(active_idx.numel())
            for b in active_idx.tolist():
                bts[b].ensure_capacity(int(seqlens[b].item()) + 1, alloc)
                generated[b].append(int(cur_tokens[b].item()))
                prompt_ids_unpadded[b] = torch.cat(
                    [prompt_ids_unpadded[b], cur_tokens[b].view(1)], dim=0
                )
                gen_counts[b] += 1
                if int(cur_tokens[b].item()) == tok.eos_token_id or gen_counts[b] >= args.max_new:
                    active[b] = False
                    bts[b].release(alloc)
            if not bool(active.any().item()):
                break

            active_idx = torch.nonzero(active, as_tuple=False).view(-1)
            seqlens[active_idx] += 1
            bt_active = [bts[i] for i in active_idx.tolist()]
            bt_mat_active = pack_block_tables(device, bt_active, args.block_tokens, seqlens[active_idx])
            logits_next = runner.decode_one(
                last_ids=cur_tokens[active_idx].view(-1, 1),
                pos_ids=(seqlens[active_idx] - 1).view(-1, 1),
                bts=bt_active,
                block_tables=bt_mat_active,
                seqlens=seqlens[active_idx],
            )
            last_logits[active_idx] = logits_next
            cur_tokens[active_idx] = sample_next(logits_next, temperature=args.temperature, top_p=args.top_p)
        dt = (time.perf_counter() - t0) * 1000.0
        decode_ms_total += dt

        if args.metrics_every > 0 and steps % args.metrics_every == 0:
            tps = decode_tokens_total / (decode_ms_total / 1000.0) if decode_ms_total > 0 else 0.0
            print(
                "[runtime] "
                f"step={steps} active={int(active.sum().item())} "
                f"prefill_ms={prefill_ms:.2f} decode_toks_per_s={tps:.2f} "
                f"free_blocks={alloc.free_count} used_blocks={alloc.used_count}"
            )

    total_ms = prefill_ms + decode_ms_total
    tps_total = decode_tokens_total / (decode_ms_total / 1000.0) if decode_ms_total > 0 else 0.0
    print(
        "[summary] "
        f"requests={B} prefill_ms={prefill_ms:.2f} total_decode_tokens={decode_tokens_total} "
        f"decode_toks_per_s={tps_total:.2f} total_ms={total_ms:.2f} "
        f"spec_accept_ratio={(spec_stats.accepted / max(1, spec_stats.proposed)):.3f}"
    )

    for i, p in enumerate(prompts):
        text = tok.decode(generated[i], skip_special_tokens=True)
        print(f"\n=== request {i} ===")
        print(p + text)


if __name__ == "__main__":
    main()
