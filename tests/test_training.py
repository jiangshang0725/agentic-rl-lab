import json

import pytest
import torch

from local_rl.evaluate import evaluate
from local_rl.train import parser, run


def toy_args(output, steps=6, extra=()):
    return parser().parse_args(
        [
            "--backend",
            "toy",
            "--task",
            "toy",
            "--device",
            "cpu",
            "--steps",
            str(steps),
            "--batch-size",
            "4",
            "--group-size",
            "8",
            "--learning-rate",
            ".01",
            "--save-every",
            "3",
            "--output",
            str(output),
            *extra,
        ]
    )


def test_resume_matches_uninterrupted_training(tmp_path):
    full = run(toy_args(tmp_path / "full"))
    part = run(toy_args(tmp_path / "part", steps=3))
    resumed = run(toy_args(tmp_path / "resume", extra=["--resume", str(part)]))
    a = torch.load(full / "toy.pt", weights_only=True)
    b = torch.load(resumed / "toy.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    metrics = [
        json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["optimizer_updates"] > 0
    assert any(row["grad_norm"] > 0 for row in metrics)
    base = evaluate(str(full), str(tmp_path / "base.json"), base=True)
    trained = evaluate(str(full), str(tmp_path / "trained.json"))
    assert [r["question"] for r in base["results"]] == [r["question"] for r in trained["results"]]


@pytest.mark.parametrize(
    "objective", ["importance_sampling", "ppo", "cispo", "gspo", "dapo", "sft"]
)
def test_all_objectives_update_parameters(tmp_path, objective):
    checkpoint = run(toy_args(tmp_path / objective, steps=3, extra=["--objective", objective]))
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["updates"] > 0


def test_validation_and_output_protection(tmp_path):
    with pytest.raises(ValueError, match="group-size"):
        run(toy_args(tmp_path / "bad", extra=["--group-size", "1"]))
    run(toy_args(tmp_path / "used", steps=1))
    with pytest.raises(ValueError, match="not empty"):
        run(toy_args(tmp_path / "used", steps=1))


def test_resume_rejects_changed_objective(tmp_path):
    part = run(toy_args(tmp_path / "part", steps=3))
    with pytest.raises(ValueError, match="objective"):
        run(toy_args(tmp_path / "new", extra=["--resume", str(part), "--objective", "gspo"]))


@pytest.mark.parametrize("objective", ["grpo", "dapo"])
def test_no_signal_does_not_fake_optimizer_updates(tmp_path, monkeypatch, objective):
    from local_rl.policy import Action, ToyPolicy, Trajectory

    def wrong(self, row, task, args, greedy=False):
        return Trajectory([Action(list(row.operands), [9])], "invalid answer")

    monkeypatch.setattr(ToyPolicy, "rollout", wrong)
    checkpoint = run(toy_args(tmp_path / objective, steps=1, extra=["--objective", objective]))
    assert json.loads((checkpoint / "manifest.json").read_text())["updates"] == 0
    metrics = json.loads((tmp_path / objective / "metrics.jsonl").read_text())
    assert metrics["attempted_groups"] == (8 if objective == "dapo" else 4)
