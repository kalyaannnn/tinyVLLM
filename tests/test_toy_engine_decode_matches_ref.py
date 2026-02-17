import pytest
import torch

from vllm_scratch.runtime.toy_engine import ToyEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_toy_engine_decode_matches_ref() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)

    eng = ToyEngine(
        num_blocks=2048,
        block_tokens=16,
        H=8,
        D=64,
        device=device,
        dtype=torch.float16,
    )

    s1 = eng.new_sequence()
    s2 = eng.new_sequence()
    seqs = [s1, s2]

    _ = eng.prefill(seqs, [33, 7], use_triton_v0=True)

    q_next, out_tri = eng.decode_step(seqs)
    out_ref = eng.decode_step_ref(q_next, seqs)

    torch.testing.assert_close(out_tri, out_ref, rtol=2e-2, atol=2e-2)
