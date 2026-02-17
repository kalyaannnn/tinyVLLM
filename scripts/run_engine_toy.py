import torch

from vllm_scratch.runtime.engine import EngineConfig, TinyEngine


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    eng = TinyEngine(EngineConfig(num_blocks=8192, block_tokens=16, H=8, D=64), device=device)

    r1 = eng.submit(prompt_len=33, max_new_tokens=10)
    r2 = eng.submit(prompt_len=7, max_new_tokens=25)
    r3 = eng.submit(prompt_len=49, max_new_tokens=12)

    step = 0
    while True:
        step += 1
        done, out = eng.step()
        print(f"step={step} active={len(eng.active)} done={done} out_shape={tuple(out.shape)}")
        if not eng.active:
            break

    print("all done")


if __name__ == "__main__":
    main()
