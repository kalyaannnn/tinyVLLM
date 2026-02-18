from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_scratch.hf.rope_gqa import apply_rope, build_rope_cos_sin, repeat_kv


def _reshape_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    b, t, c = x.shape
    d = c // n_head
    return x.view(b, t, n_head, d)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    ap.add_argument("--prompt", type=str, default="tiny rope test")
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--tol", type=float, default=8e-2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    bsz, seqlen = input_ids.shape
    if bsz != 1:
        raise ValueError("Expected batch=1")

    layers = model.model.layers
    layer0 = layers[0]
    attn = layer0.self_attn

    hidden = model.model.embed_tokens(input_ids)
    x = layer0.input_layernorm(hidden)

    q = _reshape_heads(attn.q_proj(x), int(attn.num_heads))
    k = _reshape_heads(attn.k_proj(x), int(attn.num_key_value_heads))
    v = _reshape_heads(attn.v_proj(x), int(attn.num_key_value_heads))

    pos = torch.arange(0, seqlen, device=device).unsqueeze(0)
    cos, sin = build_rope_cos_sin(pos, q.shape[-1], base=float(getattr(attn, "rope_theta", 10000.0)))
    q_r, k_r = apply_rope(q, k, cos, sin)
    k_rep = repeat_kv(k_r, int(attn.num_heads) // int(attn.num_key_value_heads))
    v_rep = repeat_kv(v, int(attn.num_heads) // int(attn.num_key_value_heads))

    scores = torch.einsum("bthd,bshd->bhts", q_r.float(), k_rep.float()) * (q.shape[-1] ** -0.5)
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("bhts,bshd->bthd", probs, v_rep.float()).reshape(bsz, seqlen, -1).to(dtype)
    out_ref = attn.o_proj(out_ref)

    out_hf, _, _ = attn(
        hidden_states=x,
        attention_mask=None,
        position_ids=pos,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    )
    err = (out_hf - out_ref).abs().max().item()
    print(f"[rope-gqa] max_abs_err={err:.6f} model={args.model} dtype={args.dtype}")
    assert err < args.tol, "RoPE/GQA attention mismatch"

    logits_hf = model(input_ids).logits
    print(f"[logits] shape={tuple(logits_hf.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
