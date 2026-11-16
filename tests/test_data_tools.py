import json

import pytest

from local_rl.data import load_examples, numeric_answer, reward
from local_rl.tools import calculate, tool_response


@pytest.mark.parametrize("task", ["toy", "arithmetic", "calculator", "retrieval"])
def test_train_eval_disjoint_and_deterministic(task):
    train, evaluation = load_examples(task, "train"), load_examples(task, "eval")
    assert not {r.question for r in train} & {r.question for r in evaluation}
    assert train == load_examples(task, "train")


def test_numeric_reward_uses_last_box_and_rejects_nonfinite():
    assert reward(r"\boxed{2} actually \boxed{1,000.0}", "1000") == 1
    for text in ("1000", r"\boxed{NaN}", r"\boxed{1+1}", r"\boxed{inf}", r"\boxed{10 kg}"):
        assert numeric_answer(text) is None
    with pytest.raises(ValueError):
        reward("", "NaN")


def test_jsonl_rejects_leakage(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"question": "same", "answer": 1, "split": s}) for s in ["train", "eval"]
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_examples("jsonl", "train", str(data))


def test_calculator_has_no_code_execution():
    assert calculate("(2 + 3) * -4 / 2") == "-10"
    for expression in ("__import__('os').system('id')", "2 ** 99999", "[1][0]", "1e100", "True"):
        with pytest.raises(ValueError):
            calculate(expression)
    assert "error" in tool_response("<calc>1/0</calc>", "calculator")
    assert (
        tool_response("<search>Station-003</search>", "retrieval")
        == "Station-003: access code 1051"
    )
