"""Paired greedy evaluation on the same fixed held-out questions."""

import argparse
import json
import random
from pathlib import Path

import torch

from .checkpoint import read_manifest
from .data import load_examples, numeric_answer, reward
from .policy import HFPolicy, ToyPolicy


def evaluate(
    checkpoint: str,
    output: str,
    limit: int = 20,
    device: str | None = None,
    offline: bool = False,
    base: bool = False,
):
    if limit <= 0:
        raise ValueError("--limit must be positive")
    destination = Path(output)
    if destination.exists():
        raise ValueError("Evaluation output exists; choose a new path")
    manifest = read_manifest(checkpoint)
    config = manifest["config"]
    args = argparse.Namespace(**config)
    args.device = device or args.device
    args.offline = args.offline or offline
    args.model_revision = manifest.get("model_revision") or config.get("model_revision")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select --device cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = load_examples(args.task, "eval", args.data)[:limit]
    if args.backend == "toy":
        policy = ToyPolicy(args.device)
        if not base:
            policy.restore(Path(checkpoint))
    else:
        policy = HFPolicy(args, None if base else Path(checkpoint), trainable=False)
    results = []
    for row in rows:
        sample = policy.rollout(row, args.task, args, greedy=True)
        results.append(
            {
                "question": row.question,
                "answer": row.answer,
                "text": sample.text,
                "correct": bool(reward(sample.text, row.answer)),
                "formatted": numeric_answer(sample.text) is not None,
                "truncated": sample.truncated,
                "observations": sample.observations,
            }
        )
    summary = {
        "checkpoint": checkpoint,
        "base": base,
        "split": "eval",
        "count": len(results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "format_rate": sum(r["formatted"] for r in results) / len(results),
        "truncated_fraction": sum(r["truncated"] for r in results) / len(results),
        "decoding": "greedy",
        "results": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--offline", action="store_true")
    p.add_argument(
        "--base", action="store_true", help="Use the same initial policy and evaluation questions"
    )
    evaluate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
