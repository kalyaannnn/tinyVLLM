from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_scratch.attention.reference import paged_attention_prefill_ref
from vllm_scratch.attention.triton_decode import paged_attention_decode_triton
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache


def _shape_qkv_gpt2(x: torch.Tensor, n_head: int) -> torch.Tensor:
    # x: [B,T,n_embd] -> [B,T,H,D]
    B, T, C = x.shape
    assert C % n_head == 0
    d = C // n_head
    return x.view(B, T, n_head, d)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="distilgpt2")
    ap.add_argument("--prompt", type=str, default="Hello from tinyVLLM.")
    ap.add_argument("--block_tokens", type=int, default=16)
    ap.add_argument("--num_blocks", type=int, default=4096)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Run on a CUDA Colab runtime.")

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

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    # Grab block0 attention
    # GPT2-like: model.transformer.h[0].attn
    block0 = model.transformer.h[0]
    attn = block0.attn
    n_head = attn.num_heads if hasattr(attn, "num_heads") else attn.n_head
    n_embd = model.config.n_embd
    head_dim = n_embd // n_head

    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    B, T = input_ids.shape
    assert B == 1, "keep B=1 for first integration step"

    # Build hidden_states exactly like GPT2 does before block0
    pos = torch.arange(0, T, device=device).unsqueeze(0)
    hs = model.transformer.wte(input_ids) + model.transformer.wpe(pos)
    hs = model.transformer.drop(hs)

    # Attention input to block0
    x = block0.ln_1(hs)  # [B,T,n_embd]

    # Project QKV using real weights
    qkv = attn.c_attn(x)  # [B,T,3*n_embd]
    q, k, v = qkv.split(n_embd, dim=2)

    q = _shape_qkv_gpt2(q, n_head)  # [B,T,H,D]
    k = _shape_qkv_gpt2(k, n_head)
    v = _shape_qkv_gpt2(v, n_head)

    # ----- Baseline prefill (contiguous) -----
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale  # [B,H,T,T]
    causal = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out_base = torch.einsum("bhts,bshd->bthd", probs, v.float())  # [B,T,H,D]

    # Project like GPT2Attention does
    out_base_merged = out_base.reshape(B, T, n_embd).to(dtype)
    out_base_proj = attn.c_proj(out_base_merged)  # [B,T,n_embd]

    # ----- Paged prefill (your oracle) -----
    alloc = BlockAllocator(num_blocks=args.num_blocks)
    bt = BlockTable(block_tokens=args.block_tokens)
    bt.ensure_capacity(T, alloc)

    # kv cache (layer=0 only)
    kv = PagedKVCache(
        num_layers=1,
        num_blocks=args.num_blocks,
        block_tokens=args.block_tokens,
        num_heads=n_head,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )

    kv.write_range(layer=0, bt=bt, start_t=0, k=k[0].contiguous(), v=v[0].contiguous())

    max_blocks = (T + args.block_tokens - 1) // args.block_tokens
    block_tables = torch.full((1, max_blocks), -1, device=device, dtype=torch.int32)
    for i, phys in enumerate(bt.blocks):
        block_tables[0, i] = int(phys)

    seqlens = torch.tensor([T], device=device, dtype=torch.int32)

    out_paged = paged_attention_prefill_ref(
        q=q.to(dtype),
        k_cache=kv.k(0),
        v_cache=kv.v(0),
        block_tables=block_tables,
        seqlens=seqlens,
        block_tokens=args.block_tokens,
    ).float()  # [B,T,H,D]

    out_paged_proj = attn.c_proj(out_paged.reshape(B, T, n_embd).to(dtype))

    prefill_err = (out_paged_proj - out_base_proj).abs().max().item()
    print(f"[prefill] max_abs_err={prefill_err:.6f}  (T={T}, H={n_head}, D={head_dim}, dtype={args.dtype})")

    # ----- One decode step (append 1 token) -----
    # Make a "next token" (random id) and build x_next like GPT2 would for that position.
    next_id = torch.randint(low=0, high=tok.vocab_size, size=(1, 1), device=device)
    pos_next = torch.tensor([[T]], device=device)

    hs_next = model.transformer.wte(next_id) + model.transformer.wpe(pos_next)
    hs_next = model.transformer.drop(hs_next)
    x_next = block0.ln_1(hs_next)  # [1,1,n_embd]

    qkv_next = attn.c_attn(x_next)  # [1,1,3*n_embd]
    qn, kn, vn = qkv_next.split(n_embd, dim=2)
    qn = _shape_qkv_gpt2(qn, n_head)  # [1,1,H,D]
    kn = _shape_qkv_gpt2(kn, n_head)
    vn = _shape_qkv_gpt2(vn, n_head)

    # Baseline contiguous decode output for that token
    k_cat = torch.cat([k, kn], dim=1).float()  # [1,T+1,H,D]
    v_cat = torch.cat([v, vn], dim=1).float()
    q_dec = qn[:, 0].float()  # [1,H,D]
    scores_d = torch.einsum("bhd,bshd->bhs", q_dec, k_cat) * scale  # [1,H,T+1]
    probs_d = torch.softmax(scores_d, dim=-1)
    out_base_dec = torch.einsum("bhs,bshd->bhd", probs_d, v_cat)  # [1,H,D]
    out_base_dec_proj = attn.c_proj(out_base_dec.reshape(1, n_embd).to(dtype))  # [1,n_embd]

    # Paged decode: append KV into paged cache then call Triton decode
    bt.ensure_capacity(T + 1, alloc)
    kv.write_token(layer=0, bt=bt, t=T, k_t=kn[0, 0].contiguous(), v_t=vn[0, 0].contiguous())

    max_blocks2 = (T + 1 + args.block_tokens - 1) // args.block_tokens
    block_tables2 = torch.full((1, max_blocks2), -1, device=device, dtype=torch.int32)
    for i, phys in enumerate(bt.blocks):
        block_tables2[0, i] = int(phys)
    seqlens2 = torch.tensor([T + 1], device=device, dtype=torch.int32)

    out_paged_dec = paged_attention_decode_triton(
        q=qn[:, 0].to(dtype),          # [1,H,D]
        k_cache=kv.k(0),
        v_cache=kv.v(0),
        block_tables=block_tables2,
        seqlens=seqlens2,
        block_tokens=args.block_tokens,
    ).float()  # [1,H,D]
    out_paged_dec_proj = attn.c_proj(out_paged_dec.reshape(1, n_embd).to(dtype))

    decode_err = (out_paged_dec_proj - out_base_dec_proj).abs().max().item()
    print(f"[decode]  max_abs_err={decode_err:.6f}")

    # Suggested tolerances for fp16/bf16
    if args.dtype in ("fp16", "bf16"):
        assert prefill_err < 2e-2, "prefill mismatch too large"
        assert decode_err < 2e-2, "decode mismatch too large"
    else:
        assert prefill_err < 1e-4
        assert decode_err < 1e-4

    print("OK: HF GPT2 block0 attention matches paged kernels (prefill+decode).")


if __name__ == "__main__":
    main()
