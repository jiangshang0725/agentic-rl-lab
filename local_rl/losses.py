"""Small, explicit policy objectives shared by the toy and language-model trainers."""

import torch


def distillation_terms(logps, old_logps, teacher_logps):
    """Sampled reverse-KL score-function gradient on fresh student trajectories.

    E_student[(log p_student - log p_teacher) * grad(log p_student)].
    The coefficient is detached. Only one update per rollout is supported.
    This is not full-vocabulary KL and cannot promise a stronger teacher.
    """
    return (old_logps.detach() - teacher_logps.detach()) * logps


def group_advantages(rewards: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Population std avoids NaN on constant groups; constant rewards give zero signal."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A reward group must contain at least two completions")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    centered = rewards - rewards.mean()
    if normalize:
        centered = centered / rewards.std(unbiased=False).clamp_min(1e-6)
    return centered.detach()


def policy_terms(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantage: float,
    objective: str = "grpo",
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    ref_logps: torch.Tensor | None = None,
    beta: float = 0.0,
) -> torch.Tensor:
    """One loss value per generated token; callers choose the normalization.

    GSPO uses the geometric mean ratio over one response/trajectory. Its ratio
    gradient must flow through that mean. The old policy is always detached.
    This function deliberately does not silently clamp overflowing ratios.
    """
    if logps.ndim != 1 or logps.numel() == 0 or logps.shape != old_logps.shape:
        raise ValueError("Current and old log-probabilities must be nonempty aligned vectors")
    delta = logps.float() - old_logps.detach().float()
    ratio = delta.mean().exp().expand_as(delta) if objective == "gspo" else delta.exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Policy ratio overflow: lower the learning rate/update count")
    if objective == "importance_sampling":
        terms = -ratio * advantage
    elif objective == "cispo":
        terms = -ratio.clamp(0.0, 1.0 + clip_high).detach() * logps * advantage
    elif objective in {"grpo", "ppo", "gspo", "dapo"}:
        clipped = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
        terms = -torch.minimum(ratio * advantage, clipped * advantage)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    if beta:
        if ref_logps is None or ref_logps.shape != logps.shape:
            raise ValueError("KL requires aligned reference log-probabilities")
        ref_delta = ref_logps.detach().float() - logps.float()
        terms = terms + beta * (ref_delta.exp() - ref_delta - 1.0)
    if not torch.isfinite(terms).all():
        raise FloatingPointError("Non-finite policy loss")
    return terms
