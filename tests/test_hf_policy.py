"""Real Transformers + PEFT integration, with a tiny local Qwen2 fixture and no network."""

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from local_rl.data import Example
from local_rl.policy import Action, HFPolicy, Trajectory
from local_rl.train import parser, run, validate


@pytest.fixture
def model_dir(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    directory = tmp_path / "base"
    vocabulary = {"<unk>": 0, "<eos>": 1, "<pad>": 2, "user": 3, "assistant": 4}
    for word in ["What", "is", "2", "+", "3", "?", "<calc>2+3</calc>", "\\boxed{5}"]:
        vocabulary[word] = len(vocabulary)
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    fast.chat_template = (
        "{% for m in messages %}{{m['role'] + ' ' + m['content'] + ' '}}{% endfor %}assistant "
    )
    fast.save_pretrained(directory)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    model.save_pretrained(directory, safe_serialization=True)
    return directory


def hf_args(model_dir, output, extra=()):
    args = parser().parse_args(
        [
            "--model",
            str(model_dir),
            "--device",
            "cpu",
            "--offline",
            "--steps",
            "2",
            "--save-every",
            "2",
            "--output",
            str(output),
            *extra,
        ]
    )
    validate(args)
    return args


def test_token_alignment_and_adapter_reference(model_dir, tmp_path):
    args = hf_args(model_dir, tmp_path / "run")
    policy = HFPolicy(args)
    sample = Trajectory([Action([3, 5, 6], [7, 1])], "example")
    policy.model.eval()
    with torch.no_grad():
        full = policy.model(torch.tensor([[3, 5, 6, 7, 1]])).logits.log_softmax(-1)
        expected = torch.stack([full[0, 2, 7], full[0, 3, 1]])
        reference = policy.logps(sample, reference=True).clone()
        assert torch.allclose(policy.logps(sample), expected, atol=1e-6)
    before = policy.logps(sample).detach().clone()
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.05)
    (-policy.logps(sample).mean()).backward()
    optimizer.step()
    assert not torch.allclose(before, policy.logps(sample))
    assert torch.allclose(reference, policy.logps(sample, reference=True), atol=1e-6)


def test_tool_observations_are_context_not_targets(model_dir, tmp_path, monkeypatch):
    args = hf_args(model_dir, tmp_path / "run")
    policy = HFPolicy(args)
    calls = iter(["<calc>2+3</calc>", "\\boxed{5}"])

    def generate(input_ids, **kwargs):
        tokens = policy.tokenizer.encode(next(calls), add_special_tokens=False) + [1]
        return torch.cat([input_ids, torch.tensor([tokens])], dim=1)

    monkeypatch.setattr(policy.model, "generate", generate)
    sample = policy.rollout(Example("What is 2 + 3?", "5"), "calculator", args)
    assert sample.observations == ["5"]
    assert len(sample.actions) == 2
    assert sample.token_count == 4  # Two generated words plus two EOS, no observation tokens.
    assert policy.logps(sample).shape == (4,)
    assert len(sample.actions[1].context) > len(sample.actions[0].context) + 2


def test_hf_sft_checkpoint_and_resume(model_dir, tmp_path):
    args = hf_args(model_dir, tmp_path / "sft", ["--objective", "sft"])
    checkpoint = run(args)
    assert (checkpoint / "adapter_model.safetensors").exists()
    restored = HFPolicy(args, checkpoint)
    sample = restored.supervised(Example("What is 2 + 3?", "5"), "arithmetic", args)
    assert torch.isfinite(restored.logps(sample)).all()
    resumed_args = hf_args(
        model_dir,
        tmp_path / "resume",
        ["--objective", "sft", "--steps", "3", "--resume", str(checkpoint)],
    )
    resumed = run(resumed_args)
    assert json.loads((resumed / "manifest.json").read_text())["updates"] == 3


def test_hf_rl_backward_with_mixed_rewards(model_dir, tmp_path, monkeypatch):
    args = hf_args(model_dir, tmp_path / "grpo", ["--beta", ".02"])
    count = 0

    def scripted_rollout(self, row, task, args, greedy=False):
        nonlocal count
        count += 1
        text = f"\\boxed{{{row.answer if count % 2 else '-999'}}}"
        token = 7 if count % 2 else 8
        return Trajectory([Action(self.encode(row, task), [token, 1])], text)

    monkeypatch.setattr(HFPolicy, "rollout", scripted_rollout)
    checkpoint = run(args)
    records = [
        json.loads(line) for line in (Path(args.output) / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[-1]["optimizer_updates"] == 2
    assert all(r["grad_norm"] > 0 for r in records)
    assert (checkpoint / "training.pt").exists()


def test_opd_teacher_is_frozen_and_student_is_trainable(model_dir, tmp_path):
    teacher_args = hf_args(
        model_dir, tmp_path / "teacher", ["--objective", "sft", "--learning-rate", ".01"]
    )
    teacher_dir = run(teacher_args)
    args = hf_args(
        model_dir,
        tmp_path / "opd",
        [
            "--objective",
            "opd",
            "--teacher-adapter",
            str(teacher_dir),
            "--group-size",
            "1",
            "--max-new-tokens",
            "4",
        ],
    )
    policy = HFPolicy(args)
    policy.load_teacher(str(teacher_dir))
    sample = Trajectory([Action([3, 5, 6], [7, 1])], "example")
    teacher_before = {
        k: p.detach().clone() for k, p in policy.model.named_parameters() if ".teacher." in k
    }
    teacher_logps = policy.teacher_logps(sample, 1.0)
    assert not teacher_logps.requires_grad
    assert all(not p.requires_grad for k, p in policy.model.named_parameters() if ".teacher." in k)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
    (-policy.logps(sample).mean()).backward()
    optimizer.step()
    assert all(
        torch.equal(teacher_before[k], p)
        for k, p in policy.model.named_parameters()
        if k in teacher_before
    )
    checkpoint = run(args)
    assert json.loads((checkpoint / "manifest.json").read_text())["updates"] == 2
