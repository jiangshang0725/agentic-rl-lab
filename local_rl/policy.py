"""Sequential rollout and differentiable scoring, with one resident model."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import Example, numeric_answer, prompt_for
from .checkpoint import read_manifest
from .tools import tool_response


@dataclass
class Action:
    context: list[int]
    tokens: list[int]


@dataclass
class Trajectory:
    actions: list[Action]
    text: str
    truncated: bool = False
    observations: list[str] = field(default_factory=list)
    reward: float = 0.0
    advantage: float = 0.0
    old_logps: torch.Tensor | None = None
    ref_logps: torch.Tensor | None = None
    teacher_logps: torch.Tensor | None = None

    @property
    def token_count(self) -> int:
        return sum(len(action.tokens) for action in self.actions)


class ToyPolicy(nn.Module):
    """Ten-action contextual bandit. Intentionally not a pretrained language model."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Linear(64, 10))
        self.to(device)
        self.dtype = torch.float32
        self.device = torch.device(device)

    def logits(self, operands: list[int]) -> torch.Tensor:
        x = F.one_hot(torch.tensor(operands, device=self.device), 10).float().flatten()
        return self.net(x)

    def rollout(self, row: Example, task: str, args, greedy: bool = False) -> Trajectory:
        if row.operands is None:
            raise ValueError("Toy policy needs integer operands")
        with torch.no_grad():
            logits = self.logits(list(row.operands)) / args.temperature
            token = int(
                logits.argmax()
                if greedy
                else torch.distributions.Categorical(logits=logits).sample()
            )
        return Trajectory([Action(list(row.operands), [token])], f"\\boxed{{{token}}}")

    def logps(self, trajectory: Trajectory, temperature: float = 1.0, reference: bool = False):
        if reference:
            raise ValueError("Toy demo uses beta=0; KL is available in the HF backend")
        action = trajectory.actions[0]
        return F.log_softmax(self.logits(action.context) / temperature, -1)[action.tokens]

    def save(self, directory: Path):
        torch.save(self.state_dict(), directory / "toy.pt")

    def restore(self, directory: Path):
        self.load_state_dict(
            torch.load(directory / "toy.pt", map_location=self.device, weights_only=True)
        )


class HFPolicy:
    def __init__(self, args, checkpoint: Path | None = None, trainable: bool = True):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(args.device)
        self.dtype = torch.float32
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer_source = str(checkpoint) if checkpoint else args.model
        revision = (
            read_manifest(checkpoint).get("model_revision") if checkpoint else args.model_revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=args.offline,
            trust_remote_code=False,
            revision=None if checkpoint else revision,
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("The local trainer needs a tokenizer with an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=args.offline,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="sdpa",
            revision=revision,
        ).to(self.device)
        self.model_revision = getattr(base.config, "_commit_hash", None)
        if checkpoint:
            self.model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=trainable)
        else:
            self.model = get_peft_model(
                base,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=args.lora_rank,
                    lora_alpha=2 * args.lora_rank,
                    lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
        # Stochastic dropout would change old/current probabilities independently of updates.
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        if trainable and args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.config.use_cache = False

    def load_teacher(self, directory: str):
        teacher_revision = read_manifest(directory).get("model_revision")
        if teacher_revision and self.model_revision and teacher_revision != self.model_revision:
            raise ValueError("Teacher and student base model revisions differ")
        self.model.load_adapter(directory, adapter_name="teacher", is_trainable=False)
        self.model.set_adapter("default")

    def teacher_logps(self, trajectory: Trajectory, temperature: float):
        self.model.set_adapter("teacher")
        try:
            with torch.no_grad():
                return self.logps(trajectory, temperature)
        finally:
            self.model.set_adapter("default")

    def parameters(self):
        return self.model.parameters()

    def encode(self, row: Example, task: str) -> list[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(row, task)}],
            tokenize=True,
            add_generation_prompt=True,
        )

    def rollout(self, row: Example, task: str, args, greedy: bool = False) -> Trajectory:
        from transformers import GenerationConfig

        context = self.encode(row, task)
        if len(context) > args.max_prompt_tokens:
            raise ValueError(
                f"Prompt has {len(context)} tokens, above --max-prompt-tokens; filter the data or raise the limit"
            )
        actions, observations, texts = [], [], []
        budget = args.max_new_tokens
        truncated = False
        self.model.eval()
        for _ in range(args.max_turns if task in {"calculator", "retrieval"} else 1):
            count = min(budget, args.max_seq_len - len(context))
            if count <= 0:
                truncated = True
                break
            # Start with a clean generation configuration: no inherited top-k/top-p penalties.
            config = GenerationConfig(
                max_new_tokens=count,
                do_sample=not greedy,
                num_beams=1,
                temperature=args.temperature if not greedy else 1.0,
                top_p=1.0,
                top_k=50 if greedy else 0,
                repetition_penalty=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            inputs = torch.tensor([context], device=self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=config,
                )
            tokens = outputs[0, len(context) :].tolist()
            if not tokens:
                truncated = True
                break
            actions.append(Action(context.copy(), tokens))
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(text)
            context += tokens
            budget -= len(tokens)
            truncated = tokens[-1] != self.tokenizer.eos_token_id
            if numeric_answer(text) is not None:
                break
            observation = tool_response(text, task)
            if observation is None:
                break
            observations.append(observation)
            # Observations condition later actions but never become policy-loss targets.
            context += self.tokenizer.encode(
                f"\n<observation>{observation}</observation>\nContinue with a tool call or final answer.\n",
                add_special_tokens=False,
            )
        return Trajectory(actions, "\n".join(texts), truncated, observations)

    def _action_logps(self, action: Action, temperature: float) -> torch.Tensor:
        ids = torch.tensor([action.context + action.tokens], device=self.device)
        outputs = self.model(
            input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False
        )
        # The last prompt token predicts the FIRST completion token. Include EOS.
        logits = outputs.logits[0, len(action.context) - 1 :].float() / temperature
        targets = ids[0, len(action.context) :]
        return -F.cross_entropy(logits, targets, reduction="none")

    def logps(self, trajectory: Trajectory, temperature: float = 1.0, reference: bool = False):
        if not trajectory.actions:
            raise ValueError("Cannot score an empty trajectory")
        self.model.train(torch.is_grad_enabled() and not reference)
        if reference:
            # Frozen base is the reference, even when resuming an adapter. No second model.
            with self.model.disable_adapter(), torch.no_grad():
                return torch.cat([self._action_logps(a, temperature) for a in trajectory.actions])
        return torch.cat([self._action_logps(a, temperature) for a in trajectory.actions])

    def supervised(self, row: Example, task: str, args) -> Trajectory:
        context = self.encode(row, task)
        tokens = self.tokenizer.encode(f"\\boxed{{{row.answer}}}", add_special_tokens=False)
        tokens.append(self.tokenizer.eos_token_id)
        if len(context) > args.max_prompt_tokens or len(context) + len(tokens) > args.max_seq_len:
            raise ValueError("Supervised example exceeds context limits")
        return Trajectory([Action(context, tokens)], f"\\boxed{{{row.answer}}}")

    def save(self, directory: Path):
        self.model.save_pretrained(
            directory, safe_serialization=True, selected_adapters=["default"]
        )
        self.tokenizer.save_pretrained(directory)
