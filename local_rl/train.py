"""Inspectable single-device training. Run from the repository root."""

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import read_manifest, restore_training, save_checkpoint
from .data import load_examples, reward
from .losses import distillation_terms, group_advantages, policy_terms
from .policy import Action, HFPolicy, ToyPolicy, Trajectory


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["hf", "toy"], default="hf")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--model-revision", help="Optional immutable Hugging Face model revision")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument(
        "--task",
        choices=["arithmetic", "toy", "gsm8k", "jsonl", "calculator", "retrieval"],
        default="arithmetic",
    )
    p.add_argument("--data", help="JSONL file with explicit train/eval splits")
    p.add_argument(
        "--objective",
        choices=["grpo", "ppo", "importance_sampling", "cispo", "gspo", "dapo", "sft", "opd"],
        default="grpo",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Total rollout rounds, including rounds with no update",
    )
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1, help="Number of prompt groups per update")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--updates-per-batch", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-prompt-tokens", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-turns", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.0, help="Optional KL to frozen base, HF only")
    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument(
        "--clip-high",
        type=float,
        default=None,
        help="Defaults: DAPO .28, CISPO 3.0 (ratio cap 4), others .2",
    )
    p.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--dynamic-attempts", type=int, default=8, help="Max DAPO groups tried per round"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--output", default="outputs/local/run")
    p.add_argument("--resume", help="Complete checkpoint; use a fresh output directory")
    p.add_argument("--adapter", help="Start a new run from a local adapter (optimizer/RNG reset)")
    p.add_argument("--teacher-adapter", help="OPD: frozen local teacher adapter on the same base")
    p.add_argument(
        "--offline", action="store_true", help="Load cached/local model files only; use local tasks"
    )
    return p


def validate(args):
    for name in (
        "steps",
        "samples",
        "batch_size",
        "group_size",
        "updates_per_batch",
        "max_new_tokens",
        "max_prompt_tokens",
        "max_seq_len",
        "max_turns",
        "lora_rank",
        "dynamic_attempts",
        "save_every",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.group_size < 2 and args.objective not in {"sft", "opd"}:
        raise ValueError("GRPO requires --group-size >= 2")
    if args.temperature <= 0 or args.learning_rate <= 0 or args.beta < 0:
        raise ValueError("temperature/lr must be positive and beta nonnegative")
    if args.clip_high is None:
        args.clip_high = {"dapo": 0.28, "cispo": 3.0}.get(args.objective, 0.2)
    if not all(
        math.isfinite(getattr(args, name))
        for name in ("temperature", "learning_rate", "beta", "clip_low", "clip_high")
    ):
        raise ValueError("Numeric hyperparameters must be finite")
    if not 0 <= args.clip_low < 1 or args.clip_high < 0:
        raise ValueError("Invalid clipping thresholds")
    if args.max_prompt_tokens >= args.max_seq_len:
        raise ValueError("max-prompt-tokens must be smaller than max-seq-len")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Run python -m local_rl.doctor; install uv sync --extra cuda or explicitly select --device cpu"
        )
    if args.backend == "toy":
        if args.task != "toy" or args.beta:
            raise ValueError("Toy backend requires --task toy --beta 0")
        if args.adapter:
            raise ValueError("--adapter is HF-only; toy checkpoints use --resume")
    elif args.task == "toy":
        raise ValueError("Task toy requires --backend toy")
    if args.offline and args.task == "gsm8k":
        raise ValueError("Export GSM8K to split-tagged JSONL for fully offline use")
    if args.resume and args.adapter:
        raise ValueError("Choose --resume or --adapter, not both")
    if args.objective == "opd":
        if (
            args.backend != "hf"
            or not args.teacher_adapter
            or args.updates_per_batch != 1
            or args.beta
        ):
            raise ValueError(
                "OPD requires HF, --teacher-adapter, --updates-per-batch 1 and --beta 0"
            )
    elif args.teacher_adapter:
        raise ValueError("--teacher-adapter is only used by --objective opd")
    if args.dynamic_attempts < args.batch_size and args.objective == "dapo":
        raise ValueError("dynamic-attempts must be >= batch-size")


def run(args):
    validate(args)
    rows = load_examples(args.task, "train", args.data)[: args.samples]
    data_hash = hashlib.sha256(
        json.dumps([asdict(row) for row in rows], sort_keys=True).encode()
    ).hexdigest()
    previous = read_manifest(args.resume) if args.resume else None
    if previous:
        mutable = {"steps", "output", "resume", "offline", "save_every"}
        for key, value in previous["config"].items():
            if key not in mutable and getattr(args, key) != value:
                raise ValueError(f"Resume config mismatch: {key}; use the original settings")
        if previous["data_hash"] != data_hash or args.steps <= previous["step"]:
            raise ValueError("Resume needs identical data and a larger total --steps")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output to preserve runs")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = Path(args.resume or args.adapter) if args.resume or args.adapter else None
    if args.adapter:
        adapter_config = read_manifest(args.adapter)["config"]
        if adapter_config["backend"] != "hf" or adapter_config["model"] != args.model:
            raise ValueError("Adapter base model mismatch")
    policy = ToyPolicy(args.device) if args.backend == "toy" else HFPolicy(args, checkpoint)
    if args.teacher_adapter:
        teacher_config = read_manifest(args.teacher_adapter)["config"]
        if teacher_config["backend"] != "hf" or teacher_config["model"] != args.model:
            raise ValueError("Teacher and student must use the same base model/tokenizer")
        policy.load_teacher(args.teacher_adapter)
    if args.backend == "toy" and checkpoint:
        policy.restore(checkpoint)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.device == "cuda" and policy.dtype == torch.float16
    )
    start, updates = (previous["step"], previous["updates"]) if previous else (0, 0)
    if previous:
        restore_training(args.resume, optimizer, scaler, args.device)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(
        f"Trainable parameters: {sum(p.numel() for p in parameters):,}; examples: {len(rows)}",
        flush=True,
    )
    last_checkpoint = None
    for step in range(start, args.steps):
        began = time.perf_counter()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        groups, all_samples = [], []
        attempts = args.dynamic_attempts if args.objective == "dapo" else args.batch_size
        used_attempts = 0
        for _ in range(attempts):
            used_attempts += 1
            row = random.choice(rows)
            if args.objective == "sft":
                if args.backend == "toy":
                    sample = Trajectory(
                        [Action(list(row.operands), [int(row.answer)])], f"\\boxed{{{row.answer}}}"
                    )
                else:
                    sample = policy.supervised(row, args.task, args)
                group = [sample]
            else:
                group = [policy.rollout(row, args.task, args) for _ in range(args.group_size)]
                for sample in group:
                    sample.reward = reward(sample.text, row.answer)
                    if args.objective == "dapo" and sample.truncated:
                        # Teaching variant: discard capped trajectories via zero reward.
                        sample.reward = 0.0
                advantages = (
                    torch.zeros(len(group))
                    if args.objective == "opd"
                    else group_advantages(
                        torch.tensor([s.reward for s in group]), args.normalize_advantages
                    )
                )
                for sample, advantage in zip(group, advantages):
                    sample.advantage = float(advantage)
                    with torch.no_grad():
                        sample.old_logps = policy.logps(sample, args.temperature).detach().cpu()
                        if args.objective == "opd":
                            sample.teacher_logps = (
                                policy.teacher_logps(sample, args.temperature).detach().cpu()
                            )
                        if args.beta:
                            sample.ref_logps = (
                                policy.logps(sample, args.temperature, reference=True)
                                .detach()
                                .cpu()
                            )
            for sample in group:
                all_samples.append(
                    {
                        "step": step + 1,
                        "question": row.question,
                        "answer": row.answer,
                        "text": sample.text,
                        "reward": sample.reward,
                        "advantage": sample.advantage,
                        "tokens": sample.token_count,
                        "truncated": sample.truncated,
                        "observations": sample.observations,
                    }
                )
            mixed = any(s.advantage != 0 for s in group)
            if args.objective != "dapo" or mixed:
                groups.append(group)
            if len(groups) == args.batch_size:
                break
        selected = [sample for group in groups for sample in group]
        active_groups = sum(any(s.advantage != 0 for s in group) for group in groups)
        if args.objective in {"sft", "opd"}:
            active_groups = len(groups)
        if args.objective not in {"sft", "opd"} and not active_groups and not args.beta:
            selected = []
        total_loss = grad_norm = 0.0
        # DAPO may exhaust its bounded resampling budget. Log a skipped update honestly.
        if selected:
            token_denominator = sum(s.token_count for s in selected)
            for _ in range(args.updates_per_batch):
                optimizer.zero_grad(set_to_none=True)
                iteration_loss = 0.0
                for sample in selected:
                    logps = policy.logps(
                        sample, 1.0 if args.objective == "sft" else args.temperature
                    )
                    if args.objective == "sft":
                        terms = -logps
                    elif args.objective == "opd":
                        terms = distillation_terms(
                            logps,
                            sample.old_logps.to(args.device),
                            sample.teacher_logps.to(args.device),
                        )
                    else:
                        terms = policy_terms(
                            logps,
                            sample.old_logps.to(args.device),
                            sample.advantage,
                            args.objective,
                            args.clip_low,
                            args.clip_high,
                            sample.ref_logps.to(args.device)
                            if sample.ref_logps is not None
                            else None,
                            args.beta,
                        )
                    # GRPO/GSPO: response mean. DAPO/SFT: mean over all selected action tokens.
                    loss = (
                        terms.sum() / token_denominator
                        if args.objective in {"dapo", "sft", "opd"}
                        else terms.mean() / len(selected)
                    )
                    scaler.scale(loss).backward()  # Release each trajectory graph immediately.
                    iteration_loss += float(loss.detach())
                scaler.unscale_(optimizer)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                )
                scaler.step(optimizer)
                scaler.update()
                updates += 1
                total_loss += iteration_loss / args.updates_per_batch
        rewards = [s["reward"] for s in all_samples]
        metrics = {
            "step": step + 1,
            "optimizer_updates": updates,
            "loss": total_loss,
            "reward": sum(rewards) / len(rewards),
            "active_groups": active_groups,
            "attempted_groups": used_attempts,
            "grad_norm": grad_norm,
            "mean_tokens": sum(s["tokens"] for s in all_samples) / len(all_samples),
            "truncated_fraction": sum(s["truncated"] for s in all_samples) / len(all_samples),
            "seconds": time.perf_counter() - began,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
            if args.device == "cuda"
            else 0,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
            if args.device == "cuda"
            else 0,
        }
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, allow_nan=False) + "\n")
        with (output / "rollouts.jsonl").open("a", encoding="utf-8") as file:
            for record in all_samples:
                file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(metrics), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            last_checkpoint = save_checkpoint(
                policy, optimizer, scaler, args, step + 1, updates, data_hash
            )
    print(f"Saved checkpoint: {last_checkpoint}", flush=True)
    return last_checkpoint


def main():
    try:
        run(parser().parse_args())
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "GPU out of memory. Close GPU applications; reduce --max-new-tokens and --max-seq-len; use the 0.5B model. No remote fallback is used."
        ) from exc


if __name__ == "__main__":
    main()
