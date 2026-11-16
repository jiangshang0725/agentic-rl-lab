"""Deterministic train/eval partitions and strict, local numeric rewards."""

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


@dataclass(frozen=True)
class Example:
    question: str
    answer: str
    operands: tuple[int, int] | None = None


def numeric_answer(text: str) -> Decimal | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not matches:
        return None
    value = matches[-1].strip().replace(",", "")
    # Reject units, expressions, infinities, NaNs and arbitrary executable text.
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def reward(text: str, answer: str) -> float:
    parsed = numeric_answer(text)
    try:
        target = Decimal(answer.replace(",", "").strip())
    except InvalidOperation as exc:
        raise ValueError(f"Expected a numeric ground truth, got {answer!r}") from exc
    if not target.is_finite():
        raise ValueError("Ground truth must be finite")
    return float(parsed is not None and parsed == target)


def load_examples(task: str, split: str, path: str | None = None) -> list[Example]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    if task == "gsm8k":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Install the datasets extra for GSM8K") from exc
        rows = load_dataset("openai/gsm8k", "main", split="train" if split == "train" else "test")
        return [
            Example(row["question"], row["answer"].rsplit("####", 1)[-1].strip()) for row in rows
        ]
    if task == "jsonl":
        if not path:
            raise ValueError("jsonl requires --data with question, answer and split fields")
        examples = []
        questions = {}
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"train", "eval"}:
                raise ValueError(f"Line {line_number}: split must be train or eval")
            question, answer = str(row["question"]).strip(), str(row["answer"]).strip()
            if not question or question in questions:
                raise ValueError(f"Line {line_number}: empty or duplicate question across splits")
            questions[question] = row["split"]
            reward("", answer)  # Validate numeric targets before loading any model.
            if row["split"] == split:
                examples.append(Example(question, answer))
        if not examples:
            raise ValueError(f"No {split} examples in {path}")
        return examples
    if task == "toy":
        rows = [
            Example(f"max({a}, {b})", str(max(a, b)), (a, b)) for a in range(10) for b in range(10)
        ]
    elif task in {"arithmetic", "calculator"}:
        rows = [
            Example(f"What is {a} + {b}?", str(a + b), (a, b)) for a in range(20) for b in range(20)
        ]
    elif task == "retrieval":
        rows = [
            Example(
                f"Find the access code for Station-{i:03d} in the local records.",
                str(1000 + i * 17),
            )
            for i in range(100)
        ]
    else:
        raise ValueError(f"Unknown task: {task}")
    # A fixed partition, independent of training seed and --samples, prevents overlap.
    random.Random(1729).shuffle(rows)
    cut = int(len(rows) * 0.8)
    return rows[:cut] if split == "train" else rows[cut:]


def prompt_for(example: Example, task: str) -> str:
    instruction = "Answer briefly. Put only the final number inside \\boxed{...}."
    if task == "calculator":
        instruction += " You may call a calculator using <calc>expression</calc>, then stop and wait for an observation."
    elif task == "retrieval":
        instruction += " Search local records using <search>Station-000</search> (replace the ID), then stop and wait for an observation."
    return instruction + "\n" + example.question
