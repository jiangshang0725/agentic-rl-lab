# 本地训练入门：从 Windows 环境到第一次 GRPO 更新

适用设备：**RTX 3070 8GB 显存、16GB 内存、Windows PowerShell**。代码分支：`feat/local-training-8gb`。本文对应本地版 `local_rl/`，更新于 2026-11-16。

本章替代原来的“注册 PyTRIO / SwanLab，再调用云端训练”入门流程。你将亲自运行采样、奖励计算、反向传播、参数更新和评估，**不需要付费 API、云 GPU、PyTRIO 账号或 SwanLab 账号**。依赖和公开模型首次需要下载；准备齐全后，内置任务可以离线训练。

本项目基于 [Datawhale Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab) 改造，保留上游署名与许可证。[原远程入门教程存档](../../remote/getting-started-upstream.md)仅供对照；其他历史章节尚未全部迁移。不要混用原网页中的安装和训练命令。

如果已经解压并装好环境，直接从“检查 CUDA”开始。如果只想先理解算法，完成 CPU toy 实验就能学习完整的策略更新过程。

## 目录

1. [你的电脑能做什么](#scope)
2. [从已下载的 ZIP 准备项目](#project)
3. [uv、Python 与旧环境](#environment)
4. [国内镜像与依赖安装](#mirrors)
5. [检查 CUDA](#cuda)
6. [第一课：CPU toy](#toy)
7. [第二课：下载模型与 GPU 训练检查](#model)
8. [第三课：真正运行 GRPO](#grpo)
9. [奖励、优势、损失与日志](#logs)
10. [没有学习信号时先做 SFT](#sft)
11. [独立评估](#evaluation)
12. [保存、续训与加载适配器](#checkpoints)
13. [其他算法、工具任务与 OPD](#extensions)
14. [自定义数据与 GSM8K](#data)
15. [故障排查与 8GB 显存调整](#troubleshooting)
16. [读代码、练习与完成标准](#learning)
17. [命令速查与参考资料](#reference)

<a id="scope"></a>

## 1. 你的电脑能做什么

**适合学习本地强化学习训练流程。** 起点是随机初始化的微型策略网络，以及已经预训练好的 `Qwen/Qwen2.5-0.5B-Instruct` 加 LoRA。8GB 显存不适合把原项目的大模型、长上下文、多模型同时驻留配置直接搬过来。

| 学习目标 | 本地实现 | 定位 |
| --- | --- | --- |
| 观察采样、奖励、优势与参数更新 | CPU toy，1,994 个参数 | 第一课，不需要下载模型 |
| 对真实语言模型做 SFT / GRPO | Qwen2.5-0.5B + LoRA rank 8 | 8GB GPU 起步配置，使用短回答 |
| 理解 Agent 与工具交互 | 本地计算器、合成记录检索 | 短轨迹，最多两轮起步 |
| 从零预训练大语言模型 | 当前没有提供 | 不作为这台机器的学习目标 |
| 完整复现上游全部研究实验 | 当前没有提供 | 具体范围见[迁移表](../../docs/local/migration.md) |

“本地微调”是加载已有模型，只更新 LoRA 参数；toy 才是从随机参数开始训练。LoRA 减少可训练参数与优化器状态，但基础模型、生成缓存和中间激活仍占显存。

目前已完成 Linux 实际小模型训练检查，以及 Windows / Linux CPU 自动化检查。**尚未在你的 3070 上实测峰值显存和速度**，因此先做诊断和两轮 SFT，不承诺固定训练时间。16GB 内存下先只运行一个训练进程；磁盘建议预留约 10–15GB 或更多给环境、模型缓存和日志，这不是显存测量值。

| 组件 | 运行位置 | 负责什么 |
| --- | --- | --- |
| uv | 本机 | 安装 Python、创建环境、安装依赖 |
| PyTorch | 本机 CPU / GPU | 张量、自动求导和优化器更新 |
| Transformers | 本机 | 加载模型、分词、生成回答 |
| PEFT | 本机 | 添加和保存 LoRA 适配器 |
| `local_rl` | 本机 | 数据、奖励、训练循环、评估与日志 |
| 模型下载站 | 初次准备时联网 | 提供公开权重文件，不参与后续训练计算 |

<a id="project"></a>

## 2. 从已下载的 ZIP 准备项目

按你的实际路径操作：

| 路径 | 用途 |
| --- | --- |
| `E:\Agentic_RL\agentic-rl-lab-feat-local-training-8gb.zip` | 已下载的本地版源码 |
| `E:\Agentic_RL\agentic-rl-lab` | 接下来运行的项目 |
| `E:\Agentic_RL\agentic-rl-lab-original-backup` | 旧版备份，保留即可 |

**已经解压成功就跳过解压命令**，只运行本节末尾的目录检查。这个 ZIP 已包含目标分支的代码，不必再 `git clone` 或切换分支。

尚未解压时，可用资源管理器解压，把包含 `pyproject.toml` 和 `local_rl` 的那一级文件夹改名为 `agentic-rl-lab`。也可在 PowerShell 运行下面整块；目标目录已存在时会停止，避免覆盖文件：

```powershell
Set-Location "E:\Agentic_RL"
$ErrorActionPreference = "Stop"
if (Test-Path ".\agentic-rl-lab") {
    throw "目标目录已存在。若已是本地版，直接进入；否则先给旧目录另起备份名。"
}
$unpackDir = Join-Path (Get-Location) ("unpack-" + [guid]::NewGuid().ToString("N"))
Expand-Archive -LiteralPath ".\agentic-rl-lab-feat-local-training-8gb.zip" -DestinationPath $unpackDir
$sourceDir = Join-Path $unpackDir "agentic-rl-lab-feat-local-training-8gb"
if (-not (Test-Path (Join-Path $sourceDir "local_rl\train.py"))) {
    throw "未找到本地训练代码，请检查压缩包内的根目录名称和分支。"
}
Move-Item -LiteralPath $sourceDir -Destination ".\agentic-rl-lab"
Set-Location ".\agentic-rl-lab"
```

检查实际工作目录：

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
Test-Path ".\pyproject.toml"
Test-Path ".\local_rl\train.py"
```

两行都应输出 `True`。后文都在这个目录执行，不是在 `E:\Agentic_RL` 或备份目录。

ZIP 解压目录没有 Git 历史，不能直接 `git pull`。以后更新可重新下载分支 ZIP 解压到另一个目录，或只替换明确修改的源码 / 文档。保留自己的输出、模型和镜像配置，不要搬移 `.venv`。**本次教程更新不改变训练代码或依赖，已有本地版环境无需为此重装。**

<a id="environment"></a>

## 3. uv、Python 与旧环境

| 名称 | 是什么 | 现在怎么处理 |
| --- | --- | --- |
| `uv` | 安装与管理工具 | 已安装过，通常继续用 |
| Python 3.12 | 解释器 | 本教程统一使用这一版本 |
| 项目的 `.venv` | 此目录专属的解释器入口和依赖 | 在新项目内由 `uv sync` 创建 |
| uv 缓存 | 可复用的安装包缓存 | 保留，避免重复下载 |

备份中的旧 `.venv` 不需要删除，也不建议复制到新目录。它可能记录旧路径，并包含上游 CPU Torch / 远程 SDK。**新建环境不等于重新下载所有已缓存的包。**

打开新的 PowerShell，进入项目目录，检查：

```powershell
uv --version
```

本地版安装验证使用 uv `0.13.6`。现有版本能正常读取锁文件就继续用；提示锁文件版本不支持时先更新 uv，不要删除锁文件来绕过问题。

尚未安装 uv 时，参照 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。已有系统 Python 和 `py` 启动器时，也可通过清华源安装验证过的版本：

```powershell
py -m pip install --user uv==0.13.6 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后重开终端，确认 `uv --version` 可用。没有 `py` 时使用官方安装方式，不必反复执行这条命令。

后文直接调用 `.\.venv\Scripts\python.exe`，**无需激活环境或修改 PowerShell 执行策略**。各块命令按顺序执行，出现错误先处理；`$ErrorActionPreference` 不保证外部程序失败后自动停止，安装步骤会另行检查退出码。

<a id="mirrors"></a>

## 4. 国内镜像与依赖安装

### 4.1 设置今后 uv 的默认镜像

在 PowerShell 执行一次：

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
[Environment]::SetEnvironmentVariable("UV_DEFAULT_INDEX", $env:UV_DEFAULT_INDEX, "User")

$env:UV_PYTHON_INSTALL_MIRROR = "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone/"
[Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_MIRROR", $env:UV_PYTHON_INSTALL_MIRROR, "User")
```

`$env:...` 立即作用于当前窗口；`User` 配置让以后启动的终端继承。其他已打开的终端、VS Code 可能需要重启。

| 配置 | 负责下载什么 | 不会自动处理什么 |
| --- | --- | --- |
| `UV_DEFAULT_INDEX` | 普通 Python 包 | 项目专用 Torch 索引、模型、Git 仓库 |
| `UV_PYTHON_INSTALL_MIRROR` | uv 管理的 CPython 解释器 | Python 包、CUDA 驱动、模型 |
| 下一步的 PyTorch 索引 | 指定 CPU / CUDA Torch 轮子 | 其他项目的自定义源 |

这是默认镜像，不是强制所有网络请求都经过国内地址。专用索引、直接 URL、已有锁文件需分别处理；中科大 Release 镜像未收录的文件可能回源。

### 4.2 替换本项目的 PyTorch 专用源

项目将 CPU / CUDA 12.4 Torch 分别绑定到专用索引。下面只换地址，保留版本和 extra 对应关系：

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
$uvProjectPath = Join-Path (Get-Location) "pyproject.toml"
$uvProjectText = [System.IO.File]::ReadAllText($uvProjectPath)
$uvProjectText = $uvProjectText.Replace("https://download.pytorch.org/whl/", "https://mirrors.nju.edu.cn/pytorch/whl/")
[System.IO.File]::WriteAllText($uvProjectPath, $uvProjectText, [System.Text.UTF8Encoding]::new($false))
Select-String -Path ".\pyproject.toml" -Pattern "mirrors.nju.edu.cn"
```

最后应看到 `cpu` 和 `cu124` 两个南京大学镜像地址。重复执行替换不会重复修改内容。

### 4.3 更新锁文件来源，再安装 CUDA 依赖

```powershell
uv lock --refresh --python 3.12
if ($LASTEXITCODE -ne 0) { throw "更新锁文件失败，请先检查上方错误。" }

uv sync --python 3.12 --extra cuda --frozen
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败，请先检查上方错误。" }
```

第一条重新解析镜像来源，不能加 `--frozen`；第二条按新锁文件创建 `.venv` 并安装。没有使用 `--upgrade`，不要为了换源随意升级核心依赖。

锁文件保存了具体下载地址。只设环境变量，再对旧锁文件直接 `--frozen`，仍可能访问原地址。检查来源：

```powershell
Select-String -Path ".\uv.lock" -Pattern '^source = '
```

应能看到清华与南京大学地址。已有缓存可复用，不必清空缓存。

你有 3070，正常选择 `cuda`。暂时只用 CPU 可将安装命令的 `cuda` 改为 `cpu`；之后用 GPU 再同步一次 `cuda`。**两个 extra 互斥，不要一起装。** CUDA 版 Torch 也能运行 CPU toy。

核心组合锁定为 Torch 2.6.0、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0。预编译的 CUDA 12.4 轮子不要求完整 CUDA Toolkit，但需要兼容 NVIDIA 驱动。

<a id="cuda"></a>

## 5. 检查 CUDA

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m local_rl.doctor
```

核对诊断 JSON，以下是**成功标志，不是你的机器已经测出的结果**：

| 字段 | 应观察什么 |
| --- | --- |
| `python` | `3.12.x` |
| `packages.torch` | CUDA 构建，通常带 `+cu124` |
| `torch_cuda_runtime` | `12.4`，而非 `null` |
| `cuda_available` | `true` |
| `gpu` | 你的 RTX 3070 |
| `cuda_backward` | `true`，微型 GPU 反向传播通过 |
| `gpu_free_gib` | 当前空闲显存，通常小于标称 8GB |

`datasets` 未安装是正常的，内置任务不需要它。`nvidia-smi` 的 CUDA Version 表示驱动支持能力，不要求与 Torch 的 `12.4` 字符串完全相同。

CUDA 不可用时先做 CPU toy，再按第 15 节排查。诊断成功仅证明基本 CUDA 运算可用，实际模型训练还要单独检查。

<a id="toy"></a>

## 6. 第一课：CPU toy

任务：输入两个 0–9 的整数，从 0–9 中选出最大值。例如 `(2,7)` 选 `7` 得 1 分，其他答案得 0 分。随机初始化的 1,994 参数小网络学习这个单步决策，不涉及语言建模或 LoRA。

### 6.1 训练与查看摘要

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate 0.01 --save-every 100 --output outputs/toy-01
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/toy-01
```

| 参数 | 含义 |
| --- | --- |
| `--backend toy --task toy` | 小网络与最大值任务 |
| `--device cpu` | 在 CPU 执行真实训练 |
| `--steps 100` | 100 轮采样，不保证有 100 次优化器更新 |
| `--batch-size 4` | 每轮抽取 4 个题目组，题目可能重复 |
| `--group-size 8` | 每道题采样 8 个答案 |
| `--learning-rate 0.01` | 小网络学习率，不直接照搬给 Qwen |
| `--save-every 100` | 第 100 轮保存 |
| `--output outputs/toy-01` | 独立输出目录 |

结束时应打印 `Saved checkpoint: ...checkpoint-000100`。没有组内奖励差异的轮次可以没有优化更新。

### 6.2 比较初始与训练后策略

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/toy-01/checkpoint-000100 --base --limit 20 --output outputs/toy-01/base-eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/toy-01/checkpoint-000100 --limit 20 --output outputs/toy-01/trained-eval.json
```

toy 的 `--base` 按训练 seed 重建初始网络；两次都用独立 20 题，选择概率最大的动作。

固定 seed 验证曾得到初始 `2/20`、训练后 `11/20`。这是一次小任务观察，不是你必须达到的分数，也不是语言模型能力证据。

**完成标志：** 有训练日志、检查点、实际优化次数和两份独立评估 JSON。

<a id="model"></a>

## 7. 第二课：下载模型与 GPU 训练检查

### 7.1 下载与训练是两件事

默认模型 `Qwen/Qwen2.5-0.5B-Instruct` 的公开权重可以匿名下载，不需要购买在线推理或训练服务。为区分网络故障与训练故障，先单独下载到 Hugging Face 缓存：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', max_workers=2))"
```

成功后会打印本地 snapshot 目录。缓存通常在用户目录下，请保留；LoRA 检查点不会另外打包整个基础模型。

PyPI 镜像不负责模型下载。若 `huggingface.co` 连接失败，可选用 [HF-Mirror](https://hf-mirror.com/) 的第三方下载镜像，在**当前 PowerShell**设置后重试上面的命令：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$env:HF_HUB_DISABLE_XET = "1"
```

它不是 Hugging Face 官方站点，本教程仅下载公开文件，不传账号 token。镜像也可能临时不可用。恢复默认端点可执行 `Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue`。完整缓存后，训练加 `--offline`。

也可以把完整模型放到自选目录，使用 `--model "E:/Models/Qwen2.5-0.5B-Instruct" --offline`。目录要有配置、tokenizer 和真实权重；只有一个权重文件或 Git LFS 指针不够。**改用自定义路径后，后续训练、加载适配器和续训都保持同一个 `--model` 值。** 下文统一采用默认模型名和缓存。

### 7.2 两轮 SFT：验证真实 GPU 反向传播

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --objective sft --offline --steps 2 --max-new-tokens 64 --save-every 2 --output outputs/gpu-check-01
```

SFT 直接使用本地算术标准答案，检查前向、反向、优化与保存；不依赖随机回答是否有对有错。核对 `optimizer_updates` 增加、`grad_norm` 有限且有学习信号，并生成 `checkpoint-000002/manifest.json`。

这是运行检查，不证明两轮就让模型变强。当前 Qwen + rank 8 配置在验证中有 540,672 个可训练参数。CUDA 自动选设备支持的 BF16，否则 FP16 + 梯度缩放；CPU 用 FP32。梯度检查点默认开启，无需另装 FlashAttention、DeepSpeed 或 vLLM。

<a id="grpo"></a>

## 8. 第三课：真正运行 GRPO

### 8.1 从本地算术题开始

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --steps 20 --batch-size 1 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo-01
```

默认 `--backend hf --task arithmetic --objective grpo`：真实 Qwen、内置加法题、GRPO。第 7 节的 SFT 检查点**不会自动加载**；这里从原始 Qwen 加新 LoRA 开始。

| 参数 / 默认值 | 含义 |
| --- | --- |
| `--samples 64` | 从固定训练分区取前 64 题作为采样池 |
| `--batch-size 1` | 一轮收集一个题目组 |
| `--group-size 4` | 同题生成四个候选回答 |
| `--max-new-tokens 96` | 一条轨迹的总动作 token 预算，含 EOS；工具多轮共享 |
| `--max-prompt-tokens 256` | 初始题目上下文上限 |
| `--max-seq-len 512` | 上下文与动作总长度上限 |
| `--learning-rate 2e-5` | LoRA 学习率 |
| `--lora-rank 8` | 适配器秩 |
| `--updates-per-batch 1` | 每次采样数据做一次更新 |
| `--beta 0` | 默认不加对基模的 KL 约束 |
| `--seed 42` | 训练抽题、初始化和采样随机性 |

题库是两个 0–19 整数相加，不下载 GSM8K。400 道题固定划为 320 训练题与 80 评估题；`--samples 64` 只缩小训练池，不改变评估集。

### 8.2 一轮发生了什么

1. 从训练池随机抽题，用当前策略生成四个回答。
2. 本地规则给每个回答打 0 或 1 分。
3. 比较组内奖励，计算优势。
4. 保存旧策略对已采样 token 的 log probability，再前向计算可求导的新概率。
5. 算策略损失，逐条 backward，更新 LoRA。
6. 写日志，按间隔保存检查点。

生成和反向传播逐条执行，避免同时保留整批生成缓存或计算图；没有同时加载四份模型，也没有远程求导。

**完成标志：** 保存 `checkpoint-000020`，能区分“20 轮采样”和“多少次有效更新”。零更新时先检查奖励与文本，不能把它当成已学会任务。

<a id="logs"></a>

## 9. 奖励、优势、损失与日志

### 9.1 奖励完全在本地计算

对语言模型，最终答案须有 `\boxed{数字}`。程序读取最后一个框，解析数值，与标准答案比较。

| 标准答案为 7 时的文本 | 奖励 | 原因 |
| --- | --- | --- |
| `The answer is \boxed{7}.` | 1 | 格式与数值正确 |
| `\boxed{7.0}` | 1 | 数值等价 |
| `The answer is 7.` | 0 | 没有要求的答案框 |
| `\boxed{8}` | 0 | 数值错误 |
| `\boxed{3+4}` | 0 | 最终答案不接受表达式 |

不使用付费裁判模型。开放式文章和主观解释不能直接用这个数值奖励。

### 9.2 手算组内优势

奖励 `[0,0,1,1]` 的均值为 `0.5`、总体标准差为 `0.5`：

```text
advantage = (reward - 组内均值) / max(组内总体标准差, 0.000001)
          = [-1, -1, +1, +1]
```

正优势倾向提高该回答动作的概率，负优势倾向降低。全错 `[0,0,0,0]` 或全对 `[1,1,1,1]` 的优势都为零，没有组内相对信号。

默认 `beta=0` 时，整轮没有有效组就跳过优化器更新。`--no-normalize-advantages` 可以只减均值，不除标准差；理解默认行为后再对照。

### 9.3 打开输出文件

| 文件 / 目录 | 内容 |
| --- | --- |
| `config.json` | 本次实际参数 |
| `metrics.jsonl` | 每行一轮指标 |
| `rollouts.jsonl` | 题目、参考答案、生成文本、奖励、优势与工具观察 |
| `checkpoint-000010/` | 第 10 轮检查点 |
| `checkpoint-000020/` | 第 20 轮检查点 |

```powershell
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo-01
Get-Content -Encoding UTF8 ".\outputs\qwen-grpo-01\rollouts.jsonl" -TotalCount 4
Get-Content -Encoding UTF8 ".\outputs\qwen-grpo-01\metrics.jsonl" -Tail 3
```

JSONL 每行是一个 JSON 对象。默认 batch=1、group=4 时，前四条是第一组候选。JSON 字符串会转义反斜杠，`\\boxed` 表示实际文本中的 `\boxed`。

| 指标 | 正确理解 |
| --- | --- |
| `step` | 采样轮次 |
| `optimizer_updates` | 累计优化器更新次数 |
| `reward` | 本轮尝试候选的平均奖励，不是独立评估分数 |
| `active_groups` | 策略梯度任务中有非零优势的组数；SFT / OPD 中是处理的组数 |
| `attempted_groups` | 尝试组数，DAPO 可能重采样 |
| `loss` | 优化目标数值，不等同于错误率 |
| `grad_norm` | 裁剪前梯度范数；实际更新前裁剪到 1 |
| `mean_tokens` | 平均生成动作 token 数 |
| `truncated_fraction` | 程序标记为未正常结束的比例，结合文本与长度预算判断 |
| `seconds` | 本轮采样与计算耗时，不含初始下载 / 加载及随后保存检查点 |
| `peak_allocated_mib` | 本轮 PyTorch 活跃分配的显存峰值 |
| `peak_reserved_mib` | 本轮 PyTorch 内存池保留的显存峰值 |

显存值不是整块显卡所有进程占用，另用 `nvidia-smi` 检查其他进程。CPU 显存指标为 0。轮耗时之和也不是含下载、加载和保存的总墙钟时间。

**loss 为 0 为什么也可能在训练？** 首次更新前，新旧概率比为 1，优势均值为 0，正负项数值可能抵消；不同回答的梯度通常不抵消。一起看梯度、更新次数和独立评估。

### 9.4 可选画本地曲线

```powershell
uv sync --python 3.12 --extra cuda --extra plots --frozen
if ($LASTEXITCODE -ne 0) { throw "绘图依赖安装失败。" }
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo-01 --plot
```

图在 `outputs/qwen-grpo-01/learning.png`，无需 SwanLab。以后同步依赖时保留所需 extra；只同步 `cuda` 可能移除先前可选的绘图依赖。

<a id="sft"></a>

## 10. 没有学习信号时先做 SFT

小模型可能会加法，却不遵守答案格式，或在 96 tokens 内没给最终答案。经常全零奖励时先读 rollout，再决定是否 SFT。

SFT（监督微调）直接展示“题目 → 标准回答”，优化回答 token 的交叉熵；参考答案来自本地算术程序，不购买教师调用。

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --objective sft --steps 30 --output outputs/qwen-sft-01
```

明确加载 SFT 适配器，开始新的 GRPO 实验：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --adapter outputs/qwen-sft-01/checkpoint-000030 --steps 20 --group-size 4 --max-new-tokens 96 --output outputs/qwen-sft-grpo-01
```

30 轮只是起步预算，不保证格式率足够。先评估再决定延长。SFT 日志的 `reward=0` 是占位，**不能据此判断没有学习**；看 loss、梯度和生成评估。

SFT 用参考答案教格式与解答，GRPO 比较自己生成的候选；两者都更新参数，学习信号不同。

<a id="evaluation"></a>

## 11. 独立评估

### 11.1 基模与直接 GRPO 对照

完成第 8 节后运行：

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo-01/checkpoint-000020 --base --offline --limit 20 --output outputs/qwen-grpo-01/base-eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo-01/checkpoint-000020 --offline --limit 20 --output outputs/qwen-grpo-01/trained-eval.json
```

`--base` 按检查点设置加载原始基模、跳过适配器。两次用相同评估题与长度设置，greedy 解码，不随机采样。

| 字段 | 含义 |
| --- | --- |
| `count` | 实际评估题数 |
| `accuracy` | 合法答案框且数值正确的比例 |
| `format_rate` | 能解析合法数值答案框的比例，不要求数值正确 |
| `truncated_fraction` | 被标记为未正常结束的比例 |
| `results` | 每题参考答案、生成原文、正确性与工具观察 |

### 11.2 做过 SFT 时比较三种状态

完成第 10 节后运行：

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-sft-01/checkpoint-000030 --offline --limit 20 --output outputs/qwen-sft-01/eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-sft-grpo-01/checkpoint-000020 --offline --limit 20 --output outputs/qwen-sft-grpo-01/eval.json
```

加上上一节基模结果，就有“基模 → SFT → SFT 后 GRPO”三个观测点。若改过模型、数据或生成预算，应统一条件后重测，不能混比。

`--base` 永远指**原始基模**，不自动指 GRPO 起点的 SFT 适配器；评估 SFT 起点必须单独指定其检查点。

训练与评估题目固定分割且不相交。20 题只是小样本检查，可设 `--limit 80` 评估全部算术保留题；题库不足时报告实际数量。反复用保留集调参后，它成为验证集，正式结论还需要额外未参与调参的数据。

真实 Qwen 的短程 CPU 检查只验证流程贯通：一次 4 题评估中，训练前后严格 boxed 正确率都为 `0/4`。**没有据此声称真实模型性能提升。** 详见[验证记录](../../docs/local/validation.md)。

<a id="checkpoints"></a>

## 12. 保存、续训与加载适配器

默认每 10 轮及最后一轮保存。`manifest.json` 最后写入，表示保存完成；中断时选择最后一个包含它的检查点。

| 内容 | 用途 |
| --- | --- |
| LoRA 与 tokenizer 文件 | 加载训练后策略，仍需原基模 |
| `training.pt` | 优化器、混合精度缩放与随机状态 |
| `manifest.json` | 参数、轮次、更新次数、数据指纹、依赖版本和可获得的模型 revision |

toy 保存完整小网络；Qwen 保存 LoRA，不是整个 0.5B 模型。

将第 8 节的 20 轮实验继续到总共 40 轮：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --resume outputs/qwen-grpo-01/checkpoint-000020 --steps 40 --batch-size 1 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo-continued-01
```

这会执行第 21–40 轮，使用新输出目录，最终生成 `checkpoint-000040`。没有保存的轮次需要重跑。

| 选项 | 恢复什么 | 用途 |
| --- | --- | --- |
| `--resume` | 权重、优化器、随机状态、累计更新次数 | 继续同一实验 |
| `--adapter` | 适配器权重，重建训练状态 | SFT 后 GRPO、开始新实验 |

续训不会自动补齐所有 CLI 参数，应保持原算法、学习率、模型路径、seed、数据、batch / group 等关键配置；具体以 `config.json` 为准。想换设置就开始新实验，不要修改 manifest 绕过检查。

**当前限制：** 从 `--adapter` 热启动的实验不能直接按现有 CLI 精确续训，因为配置校验要求保留该参数，而 CLI 又禁止它与 `--resume` 同时出现。此时可用最新检查点的 `--adapter` 开始新实验，但会重置优化器 / 随机状态，不应称为精确恢复。上面从基模开始的第 8 节示例不受此限制。

CPU 测试覆盖同配置续训与不中断训练逐参数一致；跨硬件、驱动或精度不保证数值相同。

<a id="extensions"></a>

## 13. 其他算法、工具任务与 OPD

这些是后续课程。先完成 toy、GPU 运行检查和一次独立评估，再逐项尝试，不要同时启动多个任务。

### 13.1 比较损失函数

| `--objective` | 教学重点 | 实现边界 |
| --- | --- | --- |
| `grpo` / `ppo` | 组内优势 + token 概率比裁剪 | 两者用同一 clipped surrogate，没有 value critic |
| `importance_sampling` | 不裁剪的概率比目标 | 看基本策略梯度方向 |
| `cispo` | 截断比率权重并停止其梯度 | 核心损失教学实现 |
| `gspo` | 整条回答的几何平均概率比 | 序列级裁剪 |
| `dapo` | 非对称裁剪、token 均值、有界重采样 | 没有完整 soft-overlong shaping |
| `sft` | 回答 token 交叉熵 | 使用标准答案，不是奖励比较 |
| `opd` | 冻结教师给学生采样 token 打分 | 同基模适配器，采样 reverse-KL 梯度估计 |

例如先在相同算术配置下跑 10 轮 GSPO：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --objective gspo --steps 10 --output outputs/gspo-01
```

同时更换 `gspo` 和输出目录为 `importance_sampling`、`cispo` 或 `dapo` 即可分别尝试。DAPO 用 `--dynamic-attempts 8` 限制每轮尝试次数，耗尽仍可能零更新；它将截断轨迹奖励置零，是教学处理，不是完整论文的长度奖励。

默认每批只更新一次，更新前新旧策略相同、比率为 1，裁剪通常不触发。观察裁剪可在**整组对照实验**统一增加 `--updates-per-batch 2`；更新预算也随之改变，不能将不同预算的差异归因于算法。

`--beta 0.02` 可加入对冻结基模的 KL 估计，通过禁用 LoRA 获取参考概率，不额外复制基模。参考不是 SFT 适配器；默认 `beta=0` 也不等于完整带 KL 的原始 GRPO 配方。

### 13.2 本地工具交互

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --task calculator --steps 10 --max-turns 2 --output outputs/calculator-01
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --task retrieval --steps 10 --max-turns 2 --output outputs/retrieval-01
```

| 任务 | 调用文本示例 | 本机工具 |
| --- | --- | --- |
| `calculator` | `<calc>12+7</calc>` | 受限算术解析器，不执行任意 Python |
| `retrieval` | `<search>Station-003</search>` | 合成记录表，不访问搜索 API |

模型生成调用，程序返回 observation，再继续回答。日志的 `observations` 保存工具结果。观察进入下一轮上下文，**不作为模型生成的 token 计算策略损失**。

小模型可能直接回答、用错标签，或看了观察仍没给答案。以实际 rollout 为准，程序不会替它补正确答案。这是短轨迹学习环境，未完整复现 Search-R1、ReTool 或 ALFWorld。

### 13.3 本地 OPD：不用大教师 API

完成第 10 节后，用 SFT 适配器作冻结教师：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --objective opd --teacher-adapter outputs/qwen-sft-01/checkpoint-000030 --steps 20 --group-size 2 --output outputs/opd-01
```

学生采样自己的回答，教师提供这些 token 的概率信息，训练更新学生。师生共享一个冻结基模，只切换适配器，不同时驻留两个基模。学生从基模加新适配器开始，教师用已有 SFT 适配器。

要求同基模、同 tokenizer、`--updates-per-batch 1`、`--beta 0`。这是 token 级采样 reverse-KL 的梯度估计，不是全词表或完整轨迹 KL 的精确计算。教师不保证比学生强，需要分别评估；OPD 的奖励仅用于观测，不决定更新。

<a id="data"></a>

## 14. 自定义数据与 GSM8K

### 14.1 离线 JSONL

每行必须含 `question`、数值 `answer` 和 `split`：

```json
{"question":"What is 1 + 2?","answer":"3","split":"train"}
{"question":"What is 7 + 8?","answer":"15","split":"eval"}
```

保存为项目内 `data/math.jsonl`，用 **UTF-8 无 BOM**。这是格式示例，正式实验需要更多不同题目，并保留独立评估题。

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --task jsonl --data data/math.jsonl --steps 20 --output outputs/custom-01
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/custom-01/checkpoint-000020 --offline --output outputs/custom-01/eval.json
```

重复题目会被拒绝，包括跨 split 重复。当前奖励只支持有限数值；证明题、文字评价或 `1/2` 这样的表达式需其他奖励或转换成支持的数值形式。续训和评估期间保留同一个数据文件。

### 14.2 再尝试 GSM8K

内置算术跑通后，按需安装 datasets：

```powershell
uv sync --python 3.12 --extra cuda --extra datasets --frozen
if ($LASTEXITCODE -ne 0) { throw "datasets 安装失败。" }
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --task gsm8k --samples 64 --steps 20 --max-new-tokens 128 --output outputs/gsm8k-01
```

首次需下载公开数据集，使用官方 train / test，不用付费 API。长题可能超过 256 prompt tokens，程序会报错而非静默丢掉题干；128 回答 tokens 也可能不够推理。先筛短题，或测量显存后调长度，不要把低分都解释成算法失败。

完全离线时，先导出带 split 的 JSONL，再用 `--task jsonl --offline`。当前程序会拒绝 `--task gsm8k --offline`，即使数据已有缓存。

仍需画图时同步保留 plots：`uv sync --python 3.12 --extra cuda --extra datasets --extra plots --frozen`。

<a id="troubleshooting"></a>

## 15. 故障排查与 8GB 显存调整

### 15.1 按报错定位

| 现象 | 检查与处理 |
| --- | --- |
| `No module named local_rl` | 回项目根目录，确认 `local_rl/train.py` 存在，使用本章 Python 路径 |
| 仍提示 PyTRIO 登录或额度 | 很可能运行了旧 `experiments/...py`；本地入口为 `-m local_rl.train` |
| `uv` 找不到 | 检查安装与 PATH，安装后重启终端 |
| 锁文件版本不支持 | 更新 uv，不要删除锁文件后任意升级所有包 |
| 换源仍从旧站下载 | 检查 lock 的 source / URL，执行第 4 节刷新锁文件 |
| 南京大学镜像临时缺包 | 看具体包与 URL，稍后重试；必要时只恢复 Torch 官方地址并重新 lock |
| Python 下载超时或回源 | 若已有 Python 3.12，可让 `--python` 指向其完整路径 |
| `torch_cuda_runtime: null` | 装了 CPU Torch 或用错环境；重新 `uv sync --python 3.12 --extra cuda --frozen` |
| runtime 非空但 CUDA 不可用 | 检查 `nvidia-smi`、驱动与设备；更新兼容驱动并重启 |
| 禁止执行 `Activate.ps1` | 本章直接调用 Python，不需激活 |
| HF 下载超时 / 429 | 网络或限速，重试或用第 7 节镜像 / 本地完整模型；不是购买训练额度 |
| `--offline` 找不到模型 | 先完整下载，保留缓存与相同模型名 |
| Windows symlink 警告 | 通常影响缓存空间效率，检查最终是否下载成功 |
| 输出目录 / 评估文件已存在 | 将 `...-01` 改为 `...-02`，保护已有结果 |
| `Resume config mismatch` | 对照原 config 保持参数；改变设置时新建实验 |
| JSONL BOM / 解码错误 | 保存 UTF-8 无 BOM；留意 Windows PowerShell 5 默认编码 |
| loss / 梯度非有限 | 保留失败日志，检查学习率、精度和改动，恢复默认小模型配置 |

裸 `uv run python ...` 可能按没有 extra 的配置同步环境。若用 uv run，应写 `uv run --extra cuda --frozen python -m local_rl.train ...`；本教程的直接 Python 写法不会隐式同步依赖。

### 15.2 CUDA out of memory

关闭其他占 GPU 的软件，只运行一个训练进程。保留 0.5B、LoRA rank 8、默认梯度检查点。在新目录用短序列先检查反向传播：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --objective sft --steps 2 --batch-size 1 --group-size 2 --max-prompt-tokens 128 --max-seq-len 256 --max-new-tokens 48 --output outputs/gpu-short-01
```

这是短序列检查配置，不保证适合所有任务。SFT 用参考答案，`max-new-tokens` 不负责裁剪监督答案；其显存取决于真实输入 / 目标长度和总长度约束。换回 GRPO 后，较小生成预算才直接限制生成长度。

`max-prompt-tokens` 必须小于 `max-seq-len`。长题超过限制应筛题或调整上限，不能无声裁掉题干。

本实现已逐条生成和 backward，降低 group-size 主要减少采样次数与耗时，不保证显著降低单条峰值。优先缩短实际序列、清理其他 GPU 占用；不要直接换成 7B 或关闭默认梯度检查点。

### 15.3 一直 reward=0

按顺序看原始回答：

1. 数字对但没答案框：先做 SFT 格式热身。
2. 回答未完成、截断多：缩短任务，或在显存允许时增回答预算。
3. 格式对但算错：任务可能太难，回到短算术和 toy。
4. 候选总一样：观察采样多样性，先保留默认 temperature。
5. 全对却不更新：没有组内差异，也是正常退化组。

不要改成“生成成功就奖励 1”，也不要拿评估题训练制造提升。能解释没有信号的原因，本身就是学习成果。

### 15.4 中断与重复运行

选最后一个有 `manifest.json` 的完整检查点；没保存的轮次会丢失。训练不能直接向非空输出目录重跑，评估也拒绝覆盖已有文件。使用新的路径，保留可比较的原始记录。

<a id="learning"></a>

## 16. 读代码、练习与完成标准

### 16.1 按执行顺序读代码

| 文件 | 要回答的问题 |
| --- | --- |
| [`local_rl/data.py`](../../local_rl/data.py) | 怎样分训练 / 评估？什么答案得分？ |
| [`local_rl/policy.py`](../../local_rl/policy.py) | 怎样生成、哪些参数更新、token 如何对齐？ |
| [`local_rl/losses.py`](../../local_rl/losses.py) | 优势、概率比、裁剪与 KL 怎么算？ |
| [`local_rl/train.py`](../../local_rl/train.py) | 何时采样、backward、step、跳过更新？ |
| [`local_rl/checkpoint.py`](../../local_rl/checkpoint.py) | 如何保存随机状态、优化器与数据指纹？ |
| [`local_rl/evaluate.py`](../../local_rl/evaluate.py) | 如何公平比较基模与适配器？ |

逐段解释见[从采样到更新](../../docs/local/walkthrough.md)。先思考：`generate()` 为什么不保留计算图，训练时却要再前向一次？

### 16.2 五个动手练习

1. 从 toy 取一组奖励，手算均值、总体标准差和优势，对照日志。
2. 固定其他条件，比较 group-size 2 与 4 的有效组比例和耗时。
3. 找到 `loss≈0` 但 `grad_norm>0` 的记录，解释标量与导数的区别。
4. 比较基模、SFT、SFT 后 GRPO 的格式率和正确率，保留原文证据。
5. 查看工具轨迹，区分模型动作与工具观察，确认观察不是策略目标。

每次只改一个因素，使用新目录。记录模型 / revision、seed、样本数、长度预算、轮次、实际更新次数与评估条件。

### 16.3 本章完成标准

- 能解释本机和模型下载站的职责，无需付费训练服务。
- 环境创建成功，Torch 的 CPU / CUDA 构建与目标一致。
- CPU toy 跑通，有训练前后独立评估。
- GPU 诊断与真实 SFT 反向传播通过，记录实际显存。
- 能从 rollout 解释奖励、优势和更新，或解释零更新原因。
- 能区分模型缓存、LoRA 检查点、优化器状态和评估文件。

达到这些标准就完成了本地流程入门，无需先追求 GSM8K 高分或完整论文复现。

<a id="reference"></a>

## 17. 命令速查与参考资料

环境安装后在项目根目录执行；重复训练 / 评估时更换输出路径。

| 想做什么 | 入口 |
| --- | --- |
| 环境诊断 | `.\.venv\Scripts\python.exe -m local_rl.doctor` |
| 查看训练参数 | `.\.venv\Scripts\python.exe -m local_rl.train --help` |
| 查看评估参数 | `.\.venv\Scripts\python.exe -m local_rl.evaluate --help` |
| 训练摘要 | `.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo-01` |
| GPU 环境 | `uv sync --python 3.12 --extra cuda --frozen` |
| CPU 环境 | `uv sync --python 3.12 --extra cpu --frozen` |

本章按当前 CLI 核对，GPU 性能以你的机器日志为准。测试与实测条件见 [README](../../README.md) 和[验证记录](../../docs/local/validation.md)。

参考资料（镜像服务可能变化）：

- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)、[环境变量](https://docs.astral.sh/uv/reference/environment/)、[配置优先级](https://docs.astral.sh/uv/concepts/configuration-files/)
- [清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [中科大 GitHub Release 镜像](https://mirrors.ustc.edu.cn/help/github-release.html)
- [南京大学 PyTorch 镜像](https://help.mirror.nju.edu.cn/pytorch/)
- [PyTorch 历史版本与 CUDA 轮子](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-0.5B-Instruct 模型页](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [HF-Mirror](https://hf-mirror.com/)，可选第三方下载镜像
- [上游在线教程](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/)，用于对照原远程版本
