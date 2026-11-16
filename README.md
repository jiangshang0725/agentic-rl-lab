# Agentic RL Lab · 本地学习版

面向 **Windows、RTX 3070 8GB、16GB 内存** 的强化学习实验。默认入口使用 PyTorch、Transformers 和 PEFT，在自己的电脑上完成采样、评分、反向传播、优化、保存和评估。**不需要 PyTRIO、付费 API、SwanLab 账号或云 GPU。**

本项目基于 [Datawhale Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab) 改造，保留上游署名、许可证和原实验资料。原始章节及结果见 [上游 README](README_UPSTREAM.md)；那些结果不是本地小模型的结果。

## 从这里开始

1. [Windows 安装与第一次训练](docs/local/start.md)
2. [从采样到更新：按代码顺序理解 GRPO](docs/local/walkthrough.md)
3. [算法、任务与原项目的对应关系](docs/local/migration.md)
4. [验证记录与边界](docs/local/validation.md)

## 最快跑通：无需下载模型的 CPU 实验

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在仓库根目录的 PowerShell 中运行：

```powershell
uv sync --python 3.12 --extra cpu
.\.venv\Scripts\python.exe -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate 0.01 --output outputs/toy
```

这是一个从随机参数开始学习十选一答案的微型策略网络，用来观察奖励、优势和梯度；它不是语言模型，也不代表大模型推理能力。依赖安装完成后不需要联网。

## 在 3070 上训练真实语言模型

```powershell
uv sync --python 3.12 --extra cuda
.\.venv\Scripts\python.exe -m local_rl.doctor
.\.venv\Scripts\python.exe -m local_rl.train --steps 20 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo
```

默认是 `Qwen/Qwen2.5-0.5B-Instruct` + LoRA rank 8，单 GPU、逐条生成与训练、总上下文上限 512 tokens。首次需要下载公开模型权重，此后可加 `--offline`。这里的“本地训练”指微调已有模型；并非在 8GB 显卡上从零预训练一个大语言模型。

默认依赖锁定在经过验证的兼容版本，CUDA 轮子使用 12.4。无需安装 PyTRIO 或 CUDA 编译工具链；需要兼容的 NVIDIA 驱动。不要用 `uv sync --extra cpu` 后直接期望 GPU 可用。使用 `uv run` 时也应保留对应的 `--extra cuda` / `--extra cpu`；下面的教程直接调用虚拟环境中的 Python，避免意外改变依赖。

## 已提供的本地能力

| 能力 | 入口/范围 |
| --- | --- |
| GRPO / PPO clipped surrogate | `--objective grpo` / `ppo`（无 value critic） |
| Importance sampling、CISPO | `--objective importance_sampling` / `cispo` |
| GSPO | `--objective gspo`，序列几何平均比率 |
| DAPO 教学变体 | `--objective dapo`，非对称裁剪、token mean、有上限的动态采样 |
| SFT 热身 | `--objective sft`，只对回答 token 计算交叉熵 |
| 本地 OPD | `--objective opd --teacher-adapter ...`，同一基模上的冻结教师适配器 |
| 数学题 | 内置算术、可选 GSM8K、显式划分的 JSONL |
| 两轮工具交互 | `--task calculator` / `retrieval`，离线工具，工具观察不参与策略损失 |
| 评估、续训、日志 | 固定 held-out 集、优化器/RNG 恢复、JSONL、可选本地图表 |

**本地路径不等同于完整迁移上游所有论文实验。** ALFWorld、TEMPO、视觉模型、大教师、Tournament-GRPO 的原实现仍保留在 `experiments/`，需要原远程环境。请看[迁移表](docs/local/migration.md)中的具体边界。

## 评估训练前后

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo/checkpoint-000020 --base --limit 20 --output outputs/base-eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo/checkpoint-000020 --limit 20 --output outputs/trained-eval.json
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo
```

两次评估使用相同的独立题目和 greedy 解码。不要用训练集奖励上涨来代替泛化评估，也不要预期 20 轮必然提高正确率。组内全对或全错时没有奖励差异，程序会记录零有效组和零策略更新。

## 测试

```powershell
uv sync --python 3.12 --extra cpu
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check local_rl tests
```

测试使用随机初始化的微型 Qwen2 + 真实 PEFT，验证计算图、token 对齐、适配器保存、冻结教师、工具观察屏蔽和断点续训；不下载模型，不调用 API。切回 GPU 使用前再次 `uv sync --extra cuda`。

## 原始远程实验

`remote/pyproject.toml` 与 `remote/uv.lock` 保存原依赖，避免其 CPU Torch 和 PyTRIO 进入本地默认环境。原命令改为从仓库根目录执行：

```powershell
uv run --project remote python experiments/grpo/01-demo-sync.py --swanlab-mode disabled
```

这条命令**仍然使用远程训练服务**；仅供对照原实验，不属于免费本地学习路径。上游历史说明中的裸 `uv run` / `uv sync` 应加 `--project remote`。
