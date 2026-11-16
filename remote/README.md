# Historical remote environment

This directory isolates the original `pyproject.toml` and `uv.lock`. They depend on PyTRIO and CPU Torch because computation happens on the remote service.

From the repository root, replace historical commands:

- `uv sync` → `uv sync --project remote`
- `uv run python experiments/...` → `uv run --project remote python experiments/...`
- Extras such as ALFWorld remain available through `--project remote --extra alfworld`.

These are historical cloud experiments, not the local learning path. For local CUDA training, use the root project and `python -m local_rl.train`. The no-paid-API guarantee applies to `local_rl`, not to old `experiments/` scripts.
