import pytest
import torch

from local_rl.losses import group_advantages, policy_terms


def test_constant_group_has_no_signal_and_no_nan():
    for rewards in ([0.0, 0.0], [1.0, 1.0]):
        assert torch.equal(group_advantages(torch.tensor(rewards)), torch.zeros(2))
    assert torch.allclose(group_advantages(torch.tensor([0.0, 1.0])), torch.tensor([-1.0, 1.0]))
    with pytest.raises(ValueError):
        group_advantages(torch.tensor([1.0]))


@pytest.mark.parametrize(
    "objective", ["grpo", "ppo", "importance_sampling", "cispo", "gspo", "dapo"]
)
def test_positive_advantage_increases_log_probability(objective):
    old = torch.tensor([-2.0, -3.0], requires_grad=True)
    current = old.detach().clone().requires_grad_()
    policy_terms(current, old, 1.0, objective).mean().backward()
    assert (current.grad < 0).all()
    assert old.grad is None


@pytest.mark.parametrize("advantage,ratio", [(1.0, 1.5), (-1.0, 0.5)])
def test_ppo_clips_the_improving_direction(advantage, ratio):
    current = torch.tensor([ratio]).log().requires_grad_()
    policy_terms(current, torch.zeros(1), advantage).sum().backward()
    assert current.grad.item() == 0


def test_gspo_is_sequence_ratio_not_token_ratio():
    delta = torch.tensor([0.4, -0.4], requires_grad=True)
    loss = policy_terms(delta, torch.zeros(2), 1.0, "gspo").mean()
    assert loss.item() == pytest.approx(-1.0)
    loss.backward()
    assert torch.allclose(delta.grad, torch.tensor([-0.5, -0.5]))


def test_kl_and_zero_advantage():
    current = torch.tensor([-2.0, -1.0], requires_grad=True)
    terms = policy_terms(current, current.detach(), 0.0, ref_logps=current.detach(), beta=0.1)
    assert torch.equal(terms, torch.zeros(2))
    terms.sum().backward()
    assert torch.equal(current.grad, torch.zeros(2))


def test_overflow_fails_loudly():
    with pytest.raises(FloatingPointError):
        policy_terms(torch.tensor([1000.0]), torch.zeros(1), 1.0)


def test_distillation_teacher_is_detached_and_gradient_has_correct_sign():
    from local_rl.losses import distillation_terms

    student = torch.tensor([-2.0], requires_grad=True)
    teacher = torch.tensor([-1.0], requires_grad=True)
    distillation_terms(student, student.detach(), teacher).sum().backward()
    assert student.grad.item() < 0
    assert teacher.grad is None
