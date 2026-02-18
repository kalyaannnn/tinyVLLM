import torch

from vllm_scratch.runtime.engine import EngineConfig, TinyEngine


@torch.no_grad()
def test_engine_releases_blocks_on_finish() -> None:
    eng = TinyEngine(
        EngineConfig(
            num_blocks=64,
            block_tokens=8,
            H=2,
            D=8,
            num_layers=1,
            dtype=torch.float32,
            metric_interval_ticks=0,
        ),
        device=torch.device("cpu"),
    )
    _ = eng.submit(prompt_len=9, max_new_tokens=3)
    while eng.active:
        eng.step()
    assert eng.alloc.free_count == eng.cfg.num_blocks


@torch.no_grad()
def test_engine_accepts_new_requests_while_decoding() -> None:
    eng = TinyEngine(
        EngineConfig(
            num_blocks=64,
            block_tokens=8,
            H=2,
            D=8,
            num_layers=1,
            dtype=torch.float32,
            metric_interval_ticks=0,
            prefill_budget_reqs=1,
        ),
        device=torch.device("cpu"),
    )
    _ = eng.submit(prompt_len=12, max_new_tokens=8)
    eng.step()
    _ = eng.submit(prompt_len=6, max_new_tokens=4)

    max_ticks = 50
    for _ in range(max_ticks):
        eng.step()
        if not eng.active:
            break
    assert not eng.active
