# Agentic RL Lab — local learning edition

A local PyTorch / Transformers / PEFT learning path targeting Windows, an RTX 3070 with 8 GB VRAM, and 16 GB RAM. No paid training, inference, reward, logging, or search API is required by the `local_rl` path.

- [Complete Windows local tutorial (Chinese)](experiments/getting-started/readme.md)
- [Start from your current setup](docs/local/start.md) / [Offline HTML](docs/local/getting-started.html)
- [Code walkthrough (Chinese)](docs/local/walkthrough.md)
- [Migration scope](docs/local/migration.md)
- [Validation and limitations](docs/local/validation.md)
- [Original upstream README](README_EN_UPSTREAM.md)

Install `uv`, then run from the repository root:

```powershell
uv sync --python 3.12 --extra cuda
.\.venv\Scripts\python.exe -m local_rl.doctor
.\.venv\Scripts\python.exe -m local_rl.train --steps 20 --output outputs/qwen-grpo
```

Linux uses `.venv/bin/python` instead. A model download is required once; `--offline` subsequently uses cached files only. The default model is Qwen2.5-0.5B-Instruct, with LoRA rank 8, sequential sampling, sequential backward passes and short contexts. GPU memory/throughput on a 3070 still need measurement on that hardware.

For a CPU-only, model-download-free educational bandit:

```bash
uv sync --python 3.12 --extra cpu
.venv/bin/python -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate .01 --output outputs/toy
```

Implemented: explicit GRPO/PPO clipped surrogate (no value critic), importance sampling, CISPO, GSPO core loss, a bounded DAPO teaching variant, SFT, same-base frozen-adapter on-policy distillation, arithmetic/GSM8K/JSONL, and short calculator/local-retrieval trajectories. Tools execute no arbitrary Python. Evaluation uses disjoint fixed questions; checkpoints store optimizer and RNG state.

The historical `experiments/` scripts are **not all ported**. Vision, ALFWorld, TEMPO and tournament research experiments retain their upstream remote dependencies and results. The original dependency project is isolated in `remote/`; historical commands require `uv run --project remote ...`.

Run `python -m pytest -q` and `ruff check local_rl tests` in the CPU environment. Tests use tiny local Transformers/PEFT fixtures and do not download models. Preserve upstream attribution and the Apache-2.0 license.
