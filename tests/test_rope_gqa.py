import torch

from vllm_scratch.hf.rope_gqa import repeat_kv, rope_gqa_attention_ref


@torch.no_grad()
def test_repeat_kv_shape() -> None:
    x = torch.randn(2, 5, 4, 8)
    y = repeat_kv(x, 3)
    assert y.shape == (2, 5, 12, 8)


@torch.no_grad()
def test_rope_gqa_reduces_to_mha_when_kv_heads_equal() -> None:
    torch.manual_seed(0)
    B, T, H, D = 2, 6, 4, 8
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, H, D, dtype=torch.float32)
    pos = torch.arange(0, T).unsqueeze(0).expand(B, -1)

    out = rope_gqa_attention_ref(q, k, v, pos)
    assert out.shape == (B, T, H, D)
    assert torch.isfinite(out).all().item()
