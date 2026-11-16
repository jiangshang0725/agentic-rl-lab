# Windows：从安装到评估

所有命令在仓库根目录执行，终端选择 PowerShell。默认针对 RTX 3070 8GB、16GB RAM，使用 Python 3.12。保留至少约 10–15GB 可用磁盘用于环境、模型缓存和日志；这是空间预算，不是显存估计。

## 1. 创建独立环境

```powershell
git clone https://github.com/jiangshang0725/agentic-rl-lab.git
cd agentic-rl-lab
# PR 尚未合并时，需要切换到该 PR 的分支。
git switch feat/local-training-8gb
uv sync --python 3.12 --extra cuda
.\.venv\Scripts\python.exe -m local_rl.doctor
```

已有仓库则先 `git fetch origin`，再切换分支。请自行保留未提交的本地改动。

`doctor` 应显示 `cuda_available: true`、显卡名称和 `cuda_backward: true`。如果为 false，先检查 `nvidia-smi` 和驱动，再确认 `torch_cuda_runtime` 不是 null。无需设置 API key。已有的上游 `.venv` 会被 uv 按新锁文件同步；远程实验单独使用 `remote/.venv`。

也可执行 `scripts/setup-local.ps1 -Backend cuda` 完成安装和诊断。Linux 将下列 `\.venv\Scripts\python.exe` 换成 `.venv/bin/python`。

## 2. 先看懂 CPU 微型实验

CUDA 版 Torch 也能运行 CPU 示例，无需再次安装：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate 0.01 --save-every 100 --output outputs/toy
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/toy
```

输入是两个 0–9 的数字，动作是选择 两个数字中较大的那个。随机初始化的小网络真正执行反向传播。它适合理解策略梯度，但不涉及语言建模、LoRA 或长推理。

## 3. 小模型 GRPO

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --steps 20 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo
```

首次从 Hugging Face 下载公开权重，不需要推理服务。网络不可用时，可先把完整模型下载到本地目录，再使用 `--model models/Qwen2.5-0.5B-Instruct --offline`。后续训练沿用同一模型路径；检查点仅保存适配器，仍需要基模。

若组内回答经常全错或格式不合规，可先 SFT 学习短答案格式，再用同一基模加载适配器：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --objective sft --steps 30 --output outputs/warmup
.\.venv\Scripts\python.exe -m local_rl.train --adapter outputs/warmup/checkpoint-000030 --steps 20 --output outputs/after-warmup
```

SFT 的参考答案来自本地算术生成器，不是付费教师。训练前后评估可比较 base、warmup、after-warmup 三个检查点；`--base` 始终指原始基模。

## 4. 阅读日志和评估

`outputs/<run>/` 中：

| 文件 | 含义 |
| --- | --- |
| `config.json` | 本次参数 |
| `metrics.jsonl` | 每轮奖励、有效组、累计优化次数、梯度、耗时、显存 |
| `rollouts.jsonl` | 每条题目、标准答案、模型回答、奖励、优势、工具观察 |
| `checkpoint-000020/` | LoRA、tokenizer、优化器、RNG、配置、数据指纹 |

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo/checkpoint-000020 --base --output outputs/base.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo/checkpoint-000020 --output outputs/trained.json
```

内置任务的 train/eval 划分固定且不相交，GSM8K 使用官方 train/test。评估始终 greedy；`--limit` 超过题库大小时报告实际数量。不要把小样本评估写成大幅提升推理能力的证据。

可选绘图：

```powershell
uv sync --extra cuda --extra plots
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo --plot
```

## 5. 断点续训

保持原参数，只增加总轮数，使用新输出目录：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --resume outputs/qwen-grpo/checkpoint-000020 --steps 40 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo-continued
```

`--steps 40` 表示总共 40 轮，即从第 21 轮继续。程序恢复优化器、随机状态和累计更新次数，并检查数据指纹及关键参数。`--adapter` 只是加载权重开始新实验，不恢复这些状态。CPU 续训有逐参数一致性测试；跨驱动、硬件、精度的数值一致性不作保证。只加载自己生成或可信的检查点。

## 6. 切换算法与工具任务

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --objective gspo --output outputs/gspo
.\.venv\Scripts\python.exe -m local_rl.train --objective dapo --dynamic-attempts 8 --output outputs/dapo
.\.venv\Scripts\python.exe -m local_rl.train --objective cispo --output outputs/cispo
.\.venv\Scripts\python.exe -m local_rl.train --task calculator --max-turns 2 --output outputs/calculator
.\.venv\Scripts\python.exe -m local_rl.train --task retrieval --max-turns 2 --output outputs/retrieval
```

工具任务是短轨迹教学环境，不是原 ReTool 的任意代码执行，也不是在线网页搜索。模型可以生成调用标签，观察结果后再次生成；观察 token 只作为上下文，不参与策略目标。小模型可能不会稳定遵守工具协议，日志会如实记录，不自动替它生成正确答案。`max-new-tokens` 是整条轨迹的动作 token 预算，包含 EOS。

OPD 先训练一个本地 SFT 教师，再训练新学生：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --objective opd --teacher-adapter outputs/warmup/checkpoint-000030 --group-size 2 --steps 20 --output outputs/opd
```

师生共享同一冻结基模与 tokenizer，只切换不同适配器，避免同时驻留两个基模。教师固定，只有学生更新。这里只支持每批一次更新的采样 reverse-KL 梯度估计；教师质量必须通过独立评估判断。

## 7. GSM8K 和自定义数据

```powershell
uv sync --extra cuda --extra datasets
.\.venv\Scripts\python.exe -m local_rl.train --task gsm8k --samples 64 --max-new-tokens 128 --output outputs/gsm8k
```

较长题目若超过 `max-prompt-tokens` 会报错，不会静默裁掉题意。先筛选短题，或在显存允许时提高上限。96/128 tokens 可能不足以完成 GSM8K 推理；记录截断比例后再判断奖励。

离线自定义 JSONL 每行包含 `question`、数值 `answer`、`split`（`train` 或 `eval`）。例如以下是格式示例，不会自动读取聊天内容：

```json
{"question":"What is 1 + 2?","answer":"3","split":"train"}
{"question":"What is 7 + 8?","answer":"15","split":"eval"}
```

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --task jsonl --data data/math.jsonl --offline --output outputs/custom
```

同一题在文件中重复会被拒绝，包括跨 split 重复。该奖励只支持数值答案，不支持证明题、分数表达式或开放式文字评价。

## 常见问题

- **显存不足**：关闭占用显卡的软件；先缩短 `max-new-tokens` 和 `max-seq-len`，保留梯度检查点；使用 0.5B。逐条处理已经把并发批量限制为 1，减 group-size 主要缩短采样时间，不一定明显降低单条峰值。
- **一直 reward=0 / active_groups=0**：检查实际回答、格式和截断。先用算术或 SFT 热身。不要改成“运行成功就奖励 1”，那会掩盖学习问题。
- **loss=0 但 grad_norm>0**：第一次更新时概率比为 1，组内优势均值为 0，标量损失可为 0；梯度仍可非零。查看参数更新及评估。
- **没有固定训练时长承诺**：生成通常是瓶颈；用日志中的 seconds 估算你机器上的成本。CPU 时间不能换算成 3070 性能。
- **HF 下载警告或权限问题**：公开模型可匿名下载，但仍受网络/速率限制；下载到完整本地目录后可离线。Windows symlink 警告通常影响缓存空间，不等于训练 API 要付费。
- **训练中断**：使用最后一个含 `manifest.json` 的完整检查点。未保存的轮次需重跑。
