"""Local checkpoints with explicit configuration, RNG and optimizer state."""

import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import torch


def save_checkpoint(
    policy, optimizer, scaler, args, step: int, updates: int, data_hash: str
) -> Path:
    destination = Path(args.output) / f"checkpoint-{step:06d}"
    destination.mkdir(parents=True, exist_ok=False)
    policy.save(destination)
    state = {
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(state, destination / "training.pt")
    manifest = {
        "config": vars(args),
        "step": step,
        "updates": updates,
        "data_hash": data_hash,
        "python": platform.python_version(),
        "model_revision": getattr(policy, "model_revision", None),
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
    }
    # Written last: a partial checkpoint has no manifest and cannot be resumed.
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def read_manifest(directory: str | Path):
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def restore_training(directory, optimizer, scaler, device):
    state = torch.load(Path(directory) / "training.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    # Adam moves moments to the parameter devices in load_state_dict.
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"] and device.startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda_rng"])
