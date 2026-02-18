import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_scratch.hf.gpt2_paged import GPT2PagedRunner
from vllm_scratch.kv_cache.allocator import BlockAllocator
from vllm_scratch.kv_cache.block_table import BlockTable
from vllm_scratch.kv_cache.paged_kv import PagedKVCache
from vllm_scratch.runtime.sampling import sample_next


def pack_block_tables(device, bts, block_tokens, max_seqlen):
    max_blocks = (max_seqlen + block_tokens - 1) // block_tokens
    bt = torch.full((len(bts), max_blocks), -1, device=device, dtype=torch.int32)
    for i, s in enumerate(bts):
        for j, phys in enumerate(s.blocks):
            bt[i, j] = int(phys)
    return bt


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.manual_seed(0)

    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    block_tokens = 16
    num_blocks = 8192

    # prompts (continuous batching demo)
    prompts = [
        "The quick brown fox",
        "In a distant future,",
        "NYU is located in",
    ]
    max_new = 30

    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    B, Tpad = input_ids.shape
    seqlens = attn_mask.sum(dim=1).to(torch.int32)  # true prompt lengths

    # allocator + block tables per request
    alloc = BlockAllocator(num_blocks=num_blocks)
    bts = [BlockTable(block_tokens=block_tokens) for _ in range(B)]
    for b in range(B):
        bts[b].ensure_capacity(int(seqlens[b].item()), alloc)

    # KV cache (all layers)
    n_layer = int(model.config.n_layer)
    n_embd = int(model.config.n_embd)
    n_head = model.transformer.h[0].attn.n_head if hasattr(model.transformer.h[0].attn, "n_head") else model.transformer.h[0].attn.num_heads
    head_dim = n_embd // n_head
    kv = PagedKVCache(
        num_layers=n_layer,
        num_blocks=num_blocks,
        block_tokens=block_tokens,
        num_heads=n_head,
        head_dim=head_dim,
        device=device,
        dtype=torch.float16,
    )

    runner = GPT2PagedRunner(model=model, kv=kv, block_tokens=block_tokens)

    # position ids for prompts (0..Tpad-1) but masked lengths handled by seqlens
    pos = torch.arange(0, Tpad, device=device).unsqueeze(0).expand(B, -1)

    max_seqlen = int(seqlens.max().item())
    block_tables = pack_block_tables(device, [bt for bt in bts], block_tokens, max_seqlen)

    # prefill (writes KV + returns logits for all prompt positions)
    logits = runner.prefill(input_ids, pos, bts, block_tables, seqlens)  # [B,Tpad,V]

    # start from last prompt token per sequence
    last_logits = logits[torch.arange(B, device=device), seqlens - 1]  # [B,V]
    next_ids = sample_next(last_logits, temperature=0.9, top_p=0.9)

    # generated token buffers
    generated = [ [] for _ in range(B) ]
    for b in range(B):
        generated[b].append(int(next_ids[b].item()))

    # decode loop
    cur_len = seqlens.clone()
    active = torch.ones((B,), device=device, dtype=torch.bool)

    for _ in range(max_new - 1):
        if not bool(active.any().item()):
            break

        # append 1 position for active sequences
        for b in range(B):
            if active[b]:
                bts[b].ensure_capacity(int(cur_len[b].item()) + 1, alloc)

        cur_len = cur_len + active.to(torch.int32)

        max_seqlen = int(cur_len.max().item())
        block_tables = pack_block_tables(device, [bt for bt in bts], block_tokens, max_seqlen)

        last_ids = next_ids.view(B, 1)
        pos_ids = (cur_len - 1).view(B, 1)  # new position index

        logits_next = runner.decode_one(last_ids, pos_ids, bts, block_tables, cur_len)  # [B,V]

        next_ids = sample_next(logits_next, temperature=0.9, top_p=0.9)

        for b in range(B):
            if active[b]:
                tid = int(next_ids[b].item())
                generated[b].append(tid)
                if tid == tok.eos_token_id:
                    active[b] = False

    # decode text
    for i, p in enumerate(prompts):
        text = tok.decode(generated[i], skip_special_tokens=True)
        print(f"\n=== request {i} ===")
        print(p + text)


if __name__ == "__main__":
    main()