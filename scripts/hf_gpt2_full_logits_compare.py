from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_scratch.attention.reference import paged_attention_prefill_ref
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache


def _get_heads(attn) -> int:
    if hasattr(attn, "num_heads"):
        return int(attn.num_heads)
    if hasattr(attn, "n_head"):
        return int(attn.n_head)
    raise AttributeError("Cannot find num_heads/n_head on attention module")


def _shape_qkv(x: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, C = x.shape
    assert C % n_head == 0
    D = C // n_head
    return x.view(B, T, n_head, D)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--prompt", type=str, default="Hello from tinyVLLM.")
    ap.add_argument("--block_tokens", type=int, default=16)
    ap.add_argument("--num_blocks", type=int, default=8192)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--tol", type=float, default=7e-2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run on a CUDA runtime.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    B, T = input_ids.shape
    assert B == 1, "Stage 2 assumes batch=1"

    # HF baseline logits
    hf_logits = model(input_ids).logits  # [1,T,vocab]

    # Common block table
    alloc = BlockAllocator(num_blocks=args.num_blocks)
    bt = BlockTable(block_tokens=args.block_tokens)
    bt.ensure_capacity(T, alloc)

    n_layer = int(model.config.n_layer)
    n_embd = int(model.config.n_embd)
    n_head = _get_heads(model.transformer.h[0].attn)
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

    max_blocks = (T + args.block_tokens - 1) // args.block_tokens
    block_tables = torch.full((1, max_blocks), -1, device=device, dtype=torch.int32)
    for i, phys in enumerate(bt.blocks):
        block_tables[0, i] = int(phys)
    seqlens = torch.tensor([T], device=device, dtype=torch.int32)

    # Manual forward (GPT-2)
    pos = torch.arange(0, T, device=device).unsqueeze(0)
    hs = model.transformer.wte(input_ids) + model.transformer.wpe(pos)
    hs = model.transformer.drop(hs)

    scale = 1.0 / math.sqrt(head_dim)

    for l in range(n_layer):
        block = model.transformer.h[l]
        attn = block.attn

        # ---- attention ----
        residual = hs
        x = block.ln_1(hs)
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        q = _shape_qkv(q, n_head)
        k = _shape_qkv(k, n_head)
        v = _shape_qkv(v, n_head)

        kv.write_range(layer=l, bt=bt, start_t=0, k=k[0].contiguous(), v=v[0].contiguous())

        attn_out = paged_attention_prefill_ref(
            q=q.to(dtype),
            k_cache=kv.k(l),
            v_cache=kv.v(l),
            block_tables=block_tables,
            seqlens=seqlens,
            block_tokens=args.block_tokens,
        )  # [1,T,H,D] dtype

        attn_out_proj = attn.c_proj(attn_out.reshape(1, T, n_embd))
        hs = residual + attn_out_proj

        # ---- mlp ----
        residual = hs
        x = block.ln_2(hs)
        mlp_out = block.mlp(x)
        hs = residual + mlp_out

    hs = model.transformer.ln_f(hs)
    my_logits = model.lm_head(hs)  # [1,T,vocab]

    err = (my_logits - hf_logits).abs().max().item()
    print(f"[logits] max_abs_err = {err:.6f} (T={T}, dtype={args.dtype})")

    if args.dtype in ("fp16", "bf16"):
        assert err < args.tol
    else:
        assert err < 1e-4

    print("OK: full logits match HF forward.")


if __name__ == "__main__":
    main()
