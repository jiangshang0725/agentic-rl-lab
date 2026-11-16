# 零基础本地训练手册：一步一步学会运行和理解 GRPO

适用设备：**RTX 3070 8GB 显存、16GB 内存、Windows PowerShell**。代码分支：`feat/local-training-8gb`。本文对应本地版 `local_rl/`，更新于 2026-11-16。

本章替代原来的“注册 PyTRIO / SwanLab，再调用云端训练”入门流程。你将亲自运行采样、奖励计算、反向传播、参数更新和评估，**不需要付费 API、云 GPU、PyTRIO 账号或 SwanLab 账号**。依赖和公开模型首次需要下载；准备齐全后，内置任务可以离线训练。

本项目基于 [Datawhale Agentic RL Lab](https://github.com/datawhalechina/agentic-rl-lab) 改造，保留上游署名与许可证。[原远程入门教程存档](../../remote/getting-started-upstream.md)仅供对照；其他历史章节尚未全部迁移。不要混用原网页中的安装和训练命令。

**不要求你已经会 Python、命令行或强化学习。** 先学会操作，再观察实验，最后理解原理。遇到不认识的词可以继续往下看，第一次不需要背公式。

### 先选择自己的起点

| 你现在的状态 | 从哪里开始 | 今天先做到哪里 |
| --- | --- | --- |
| 不知道终端是什么，也没运行过命令 | 第 0 节 | 学会打开窗口、输入命令、识别结果 |
| 已下载 ZIP，但不知道如何整理文件 | 第 2 节 | 解压正确，并通过两项文件检查 |
| 有源码，还没有新的 Python 环境 | 第 3–5 节 | 安装成功，知道 GPU 是否可用 |
| 环境已经可用，还不理解训练 | 第 6 节 | 完成 CPU toy，看到训练前后分数 |
| 已能运行 Qwen，想理解日志 | 第 9 节，再看第 11 节 | 读懂更新次数并评估 |

### 不必一次读完

把这本手册分成四次学习，每次完成后都可以停下来。这里划分的是学习目标，不是保证完成时间。

| 学习阶段 | 阅读范围 | 留下的成果 |
| --- | --- | --- |
| 第一次：准备工作 | 第 0–5 节 | 项目目录、独立环境、诊断结果 |
| 第二次：完整小实验 | 第 6 节 | 小网络、日志、两份评估文件 |
| 第三次：真实语言模型 | 第 7–11 节 | Qwen 适配器、能解释的日志与评估 |
| 第四次：按兴趣继续 | 第 12–17 节 | 续训或一个进阶实验 |

**第一次阅读可以先跳过第 12–14 节。** GRPO 跑通之前，不必同时学习 GSPO、DAPO、OPD 和 GSM8K。所有实验都在自己的电脑运行；无需先注册训练平台或配置 API Key。

阅读约定：标为 **“输入”** 的 PowerShell 方框才需要放进终端；标为 **“示例输出”** 的内容只用来对照。每个操作先确认结果，再继续下一步。离线网页里的长命令可能自动折行，复制按钮会保留完整命令。

## 目录

0. [开始之前：文件夹、终端与命令](#basics)
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

<a id="basics"></a>

## 0. 开始之前：文件夹、终端与命令

**本节目标：** 你知道在哪里输入命令，能分清“我输入的内容”和“电脑返回的结果”。现在不需要安装新软件，也不需要编写 Python。

### 0.1 先认识五样东西

| 名词 | 可以怎样理解 | 在本项目中的例子 |
| --- | --- | --- |
| 文件 | 保存一种内容的对象 | `readme.md` 是说明书，`train.py` 是程序 |
| 文件夹，也叫目录 | 用来组织文件的容器 | `E:\Agentic_RL` |
| 路径 | 文件或文件夹的地址 | `E:\Agentic_RL\agentic-rl-lab\pyproject.toml` |
| 终端 | 通过文字指令操作电脑的窗口 | 本文使用 PowerShell |
| 程序 | 电脑执行的一组指令 | Python 执行训练代码，uv 安装依赖 |

你现在阅读的网页是一份说明书。它不会自动修改电脑；只有把命令放进 PowerShell 并执行，电脑才开始做事。不要把命令输入到浏览器地址栏、搜索框或聊天框。

### 0.2 打开 PowerShell

1. 按键盘上的 Windows 键，打开开始菜单。
2. 输入 `PowerShell`。
3. 打开 **Windows PowerShell**。普通方式打开即可，不需要“以管理员身份运行”。
4. 窗口可能是蓝色或黑色，颜色不影响本教程。

你会看到类似这一行，**这是示例输出，不要复制执行**：

```text
PS C:\Users\你的用户名>
```

`PS` 表示 PowerShell；中间的路径表示当前所在位置；最后的 `>` 后面是输入命令的地方。路径不必与示例一模一样。

如果看到 `>>>`，那是 Python 交互窗口。先输入 `exit()` 并回车，返回 PowerShell，再执行本教程命令。如果看到单独的 `>>`，常见原因是上一条命令的引号或括号没闭合；按 `Ctrl+C` 取消这段未完成输入，再重新完整粘贴。

### 0.3 先练一条不会修改文件的命令

**输入：** 只复制下面这一行，粘贴到 PowerShell，按 Enter（回车）。

```powershell
Get-Location
```

**它做什么：** 显示当前文件夹。不会训练，不会安装，也不会删除文件。

**预期现象：** 打印 `Path` 及当前路径，然后再次出现 `PS ...>`。路径可能在 C 盘，这是正常的；第 2 节会进入项目目录。

复制时使用 `Ctrl+C`，粘贴到终端可用 `Ctrl+V`。注意：只有选中了教程中的文字时，`Ctrl+C` 才是复制；在正在训练的终端里按 `Ctrl+C` 通常是中断程序。

### 0.4 什么叫“在项目根目录运行”

“根目录”在这里指这一个项目最外层的主文件夹，不是整个 E 盘。进入正确目录后，应同时能看到 `pyproject.toml`、`local_rl`、`docs` 等内容。

下面先看懂，**现在不用执行**；第 2 节解压完成后才会用到：

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
```

`Set-Location` 的作用是换位置，类似在资源管理器里打开一个文件夹；`cd` 是常见简写。路径外的英文双引号也一起复制，它们让有空格的路径仍能被正确识别。

如果现在还没解压，执行上面这条会提示路径不存在。这不表示安装坏了：先完成第 2 节即可。成功换目录通常不显示“成功”字样，而是提示符里的路径改变。

### 0.5 怎么知道命令运行完了

| 你看到的情况 | 含义 | 怎么做 |
| --- | --- | --- |
| 出现新的 `PS ...>`，没有报错 | 这条命令已经返回 | 对照本节成功标志，再继续 |
| 正在显示下载百分比或训练 JSON，尚未出现提示符 | 程序仍在运行 | 等待，不重复输入下一条 |
| 没有输出，但提示符已经回来 | 有些命令成功时本来就不打印信息 | 看对应检查步骤 |
| 出现 `Traceback`、`Error`、`Exception` 等，程序退回提示符 | 通常是执行失败 | 保存报错，先停在这一步 |
| 出现 `Warning`，后面仍在正常打印训练步骤 | 警告，不等于程序已停止 | 先记录，再结合具体内容判断 |

不要把“窗口还在”当成程序仍在训练，也不要只凭红色文字判断是否完全失败。有些进度信息也会走错误输出通道，真正要看错误内容与后续是否继续运行。

### 0.6 复制命令的六条规则

1. 不复制示例中的 `PS E:\...>` 提示符，只复制真正的命令。
2. 不复制代码块外围的三个反引号，也不用输入方框上的 `powershell`。
3. 原样复制英文半角引号、减号与路径；不要改成中文引号或长横线。
4. 一条命令很长时仍是一条命令；不要自己在中间加回车。
5. 标明“整块复制”的变量或 `if` 代码块，需要完整复制，不能只复制最后一行。
6. 第一遍先用教程给出的设置，不同时更改模型、路径和训练参数。

### 0.7 从此处记住两种不同的“参数”

**命令参数**是运行程序的选项，比如 `--steps 20` 表示这次安排 20 轮。

**模型参数**是模型内部会影响回答的大量数字。训练会调整其中一部分数字。你不需要自己逐个填写它们。

同一个词在两处含义不同：把 `--steps` 从 20 改成 40 是修改运行设置；训练过程中的优化器更新才是在修改模型参数。

**本节检查：** 你能打开 PowerShell、运行 `Get-Location`，并找到返回的路径。如果能做到，就进入下一节。

<a id="scope"></a>

## 1. 你的电脑能做什么

先用一句话认识我们要做的事：**让一个模型尝试回答题目，程序检查答案，再根据检查结果调整模型。** 调整之后，用另一组题目检查它是否有所变化。

有三个容易混淆的动作：

| 动作 | 发生了什么 | 会修改模型参数吗 |
| --- | --- | --- |
| 下载模型 | 把别人训练好的权重文件保存到硬盘 | 不会 |
| 让模型回答，也叫推理 | 使用已有权重生成文字或选择动作 | 不会 |
| 训练 / 微调 | 根据损失和梯度调整可训练权重 | 会 |

每次生成的文字不同，并不自动证明模型在学习；随机采样本身也会让回答变化。后面要看 `optimizer_updates` 和独立评估。

**你的电脑适合学习本地强化学习训练流程。** 起点是随机初始化的微型策略网络，以及已经预训练好的 `Qwen/Qwen2.5-0.5B-Instruct` 加 LoRA。8GB 显存不适合把原项目的大模型、长上下文、多模型同时驻留配置直接搬过来。

| 学习目标 | 本地实现 | 定位 |
| --- | --- | --- |
| 观察采样、奖励、优势与参数更新 | CPU toy，1,994 个参数 | 第一课，不需要下载模型 |
| 对真实语言模型做 SFT / GRPO | Qwen2.5-0.5B + LoRA rank 8 | 8GB GPU 起步配置，使用短回答 |
| 理解 Agent 与工具交互 | 本地计算器、合成记录检索 | 短轨迹，最多两轮起步 |
| 从零预训练大语言模型 | 当前没有提供 | 不作为这台机器的学习目标 |
| 完整复现上游全部研究实验 | 当前没有提供 | 具体范围见[迁移表](../../docs/local/migration.md) |

“本地微调”是加载已有模型，只更新 LoRA 参数；toy 才是从随机参数开始训练。LoRA 减少可训练参数与优化器状态，但基础模型、生成缓存和中间激活仍占显存。

目前已有 Linux 实际小模型训练检查、Windows / Linux CPU 自动化检查，以及读者提供的一段 Windows CUDA 训练日志。该片段显示至少完成一次 GPU 参数更新；第 9 节会解读它。片段不等于完整 20 轮的峰值、长期稳定性或效果评估，因此仍先做诊断和两轮 SFT，不承诺固定时长。

16GB 是电脑主内存，3070 的 8GB 是显卡自己的显存，两者不能直接相加当作 24GB 显存。初学时只运行一个训练进程。硬盘另需留出约 10–15GB 或更多给环境、模型与日志；这个数字是空间预算，不是显存测量值。

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

**本节目标：** 得到一份真正解压出来的源码，知道应该在哪个文件夹运行。

### 2.1 先认清三份东西

按你的实际路径操作：

| 路径 | 用途 |
| --- | --- |
| `E:\Agentic_RL\agentic-rl-lab-feat-local-training-8gb.zip` | 已下载的本地版源码 |
| `E:\Agentic_RL\agentic-rl-lab` | 接下来运行的项目 |
| `E:\Agentic_RL\agentic-rl-lab-original-backup` | 旧版备份，保留即可 |

**已经解压成功就跳过解压命令**，只运行本节末尾的目录检查。这个 ZIP 已包含目标分支的代码，不必再 `git clone` 或切换分支。

### 2.2 推荐新手使用鼠标解压

1. 按 `Win+E` 打开文件资源管理器。
2. 点击上方地址栏，输入 `E:\Agentic_RL`，按回车。
3. 找到 `agentic-rl-lab-feat-local-training-8gb.zip`。
4. 右键压缩包，选择“全部解压缩”或你解压软件中的“解压到……”；保存到 `E:\Agentic_RL` 下。
5. 解压完成后打开新文件夹，继续进入，直到能同时看到 `pyproject.toml` 文件和 `local_rl` 文件夹。
6. **包含这两个东西的这一层**才是我们要使用的项目目录。不要在压缩包预览窗口里直接运行程序。
7. 将这一层文件夹放在 `E:\Agentic_RL` 下，并改名为 `agentic-rl-lab`。
8. 如果目标名称已存在，先检查它是否就是已经准备好的本地版；不要选择合并或覆盖一堆文件。旧版可以另起备份名，已有 `agentic-rl-lab-original-backup` 则保持原样。

最后的对应关系应是：

| 内容 | 完整位置 |
| --- | --- |
| 项目依赖说明 | `E:\Agentic_RL\agentic-rl-lab\pyproject.toml` |
| 本地训练代码 | `E:\Agentic_RL\agentic-rl-lab\local_rl\train.py` |

如果变成 `...\agentic-rl-lab\agentic-rl-lab-feat-local-training-8gb\pyproject.toml`，说明多嵌套了一层。先整理目录，再继续；不是靠反复安装依赖解决。

### 2.3 可选：用命令解压

**已经按鼠标步骤完成，就跳过这一小节。两种方法选一种即可。** 下方整块复制到 PowerShell；外围的 `& { ... }` 把操作作为一个整体执行，目标已存在时会停止，不覆盖原文件：

```powershell
& {
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
}
```

### 2.4 让电脑检查目录是否正确

**输入：** 第一行先进入目录，后两行检查文件是否存在。

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
Test-Path ".\pyproject.toml"
Test-Path ".\local_rl\train.py"
```

**预期结果：** `Set-Location` 成功时通常不打印文字；两个 `Test-Path` 都应输出 `True`。`True` 是“存在”，`False` 是“不存在”。

若有 `False`，回到上面的路径表逐层检查：是否还在 ZIP 中、是否多嵌套一层、是否拿到了原版而不是本地分支。两项都为 `True` 才继续安装。后文都在这个目录执行，不是在 `E:\Agentic_RL` 或备份目录。

“项目根目录”就是这里。以后每次新开 PowerShell，都先进入这个位置，不必重新解压。

### 2.5 ZIP 用户以后怎样更新

ZIP 解压目录没有 Git 历史，不能直接 `git pull`。以后更新可重新下载分支 ZIP 解压到另一个目录，或只替换明确修改的源码 / 文档。保留自己的输出、模型和镜像配置，不要搬移 `.venv`。**本次教程更新不改变训练代码或依赖，已有本地版环境无需为此重装。**

<a id="environment"></a>

## 3. uv、Python 与旧环境

**本节目标：** 分清“安装工具”和“项目环境”，避免把旧版环境混进来。

Python 是执行训练程序的工具；“依赖”是训练程序需要的其他软件包，例如 PyTorch。不同项目可能需要不同版本，虚拟环境把每个项目的依赖分开存放，避免互相影响。

可以把项目文件夹理解为工作台，`.venv` 是这个工作台专用的工具箱，uv 负责把合适的工具装进箱子。工具箱创建后，你调用其中的 `python.exe`，就能用对这套依赖。

| 名称 | 是什么 | 现在怎么处理 |
| --- | --- | --- |
| `uv` | 安装与管理工具 | 已安装过，通常继续用 |
| Python 3.12 | 解释器 | 本教程统一使用这一版本 |
| 项目的 `.venv` | 此目录专属的解释器入口和依赖 | 在新项目内由 `uv sync` 创建 |
| uv 缓存 | 可复用的安装包缓存 | 保留，避免重复下载 |

备份中的旧 `.venv` 不需要删除，也不建议复制到新目录。它可能记录旧路径，并包含上游 CPU Torch / 远程 SDK。**新建环境不等于重新下载所有已缓存的包。**

### 3.1 确认 uv 已经能用

打开新的 PowerShell，按第 2 节进入项目目录。**输入：**

```powershell
uv --version
```

**预期结果：** 一行以 `uv` 开头的版本信息，例如 `uv 0.13.6`，后面可能还有构建信息。

如果提示“无法将 uv 识别为……”或“不是内部或外部命令”，说明当前终端找不到它，不表示项目源码有问题。先处理安装或 PATH，再继续。PATH 可以理解为系统查找程序的位置清单；安装后新开终端通常才能继承更新。

本地版安装验证使用 uv `0.13.6`。现有版本能正常读取锁文件就继续用；提示锁文件版本不支持时先更新 uv，不要删除锁文件来绕过问题。

### 3.2 只有 uv 不可用时才看这里

尚未安装 uv 时，参照 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。已有系统 Python 和 `py` 启动器时，也可通过清华源安装验证过的版本：

```powershell
py -m pip install --user uv==0.13.6 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

安装后重开终端，确认 `uv --version` 可用。没有 `py` 时使用官方安装方式，不必反复执行这条命令。

### 3.3 以后运行程序，使用项目自己的 Python

下一节安装完成后，`.venv` 会自动出现，不需要自己手动创建同名空文件夹。

后文直接调用 `.\.venv\Scripts\python.exe`，**无需激活环境或修改 PowerShell 执行策略**。各块命令按顺序执行，出现错误先处理；`$ErrorActionPreference` 不保证外部程序失败后自动停止，安装步骤会另行检查退出码。

<a id="mirrors"></a>

## 4. 国内镜像与依赖安装

**本节目标：** 让 uv 优先从国内镜像获取文件，并在新项目下安装 GPU 版依赖。

“镜像”是软件文件的另一个下载入口。它改变从哪里下载，不会把本地训练变成云训练，也不是模型。我们要分别照顾普通 Python 包、Python 解释器和 PyTorch 的下载来源。

本节需要联网。已成功安装环境、只想继续学习的读者可以跳到第 5 节；不需要每次训练前重复刷新锁文件。

### 4.1 设置今后 uv 的默认镜像

**在哪里输入：** 当前 PowerShell 窗口。**怎么输入：** 下面整块复制，一次执行；不必把英文变量名改成自己的用户名。

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
[Environment]::SetEnvironmentVariable("UV_DEFAULT_INDEX", $env:UV_DEFAULT_INDEX, "User")

$env:UV_PYTHON_INSTALL_MIRROR = "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone/"
[Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_MIRROR", $env:UV_PYTHON_INSTALL_MIRROR, "User")
```

逐行理解：第 1 行设置普通 Python 包的地址，第 2 行保存到当前 Windows 用户；第 4 行设置 Python 解释器的下载地址，第 5 行保存到当前用户。空行只是为了好读。

`$env:...` 是环境变量，可以理解为程序启动时读取的一条设置。`User` 是固定选项，表示“当前 Windows 用户”，不是要替换的占位符。设置成功通常不显示文字，这是正常的。

检查当前窗口是否读到了设置，分别输入：

```powershell
$env:UV_DEFAULT_INDEX
$env:UV_PYTHON_INSTALL_MIRROR
```

应分别打印清华源和中科大源地址。若打印空白，说明这个窗口没有读到对应设置，重新执行上面整块。其他已打开终端、VS Code 可能需重启；设置了 `User` 后，新启动的终端应能继承。

| 配置 | 负责下载什么 | 不会自动处理什么 |
| --- | --- | --- |
| `UV_DEFAULT_INDEX` | 普通 Python 包 | 项目专用 Torch 索引、模型、Git 仓库 |
| `UV_PYTHON_INSTALL_MIRROR` | uv 管理的 CPython 解释器 | Python 包、CUDA 驱动、模型 |
| 下一步的 PyTorch 索引 | 指定 CPU / CUDA Torch 轮子 | 其他项目的自定义源 |

这是默认镜像，不是强制所有网络请求都经过国内地址。专用索引、直接 URL、已有锁文件需分别处理；中科大 Release 镜像未收录的文件可能回源。

### 4.2 替换本项目的 PyTorch 专用源

**为什么还要做这一步：** 普通包走清华源，不代表 GPU 版 PyTorch 也会跟着换源。本项目已单独指定 CPU / CUDA 12.4 Torch 的地址，需要再改这里。

下面整块会读出 `pyproject.toml`、替换下载地址，再保存。你不用手工编辑大段文件，也不需要理解每个括号。先确认两个 `Test-Path` 已通过，再整块执行：

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
$uvProjectPath = Join-Path (Get-Location) "pyproject.toml"
$uvProjectText = [System.IO.File]::ReadAllText($uvProjectPath)
$uvProjectText = $uvProjectText.Replace("https://download.pytorch.org/whl/", "https://mirrors.nju.edu.cn/pytorch/whl/")
[System.IO.File]::WriteAllText($uvProjectPath, $uvProjectText, [System.Text.UTF8Encoding]::new($false))
Select-String -Path ".\pyproject.toml" -Pattern "mirrors.nju.edu.cn"
```

**预期结果：** 最后一行会列出两处南京大学镜像地址，分别以 `/cpu` 和 `/cu124` 结尾。`cpu` 是 CPU 构建，`cu124` 是 CUDA 12.4 构建。

这段代码中的 `$uvProjectText` 是暂存文件文字的变量，不是模型；`UTF8Encoding` 是保存文字的编码方式，不影响训练算法。若报文件不存在，先检查当前目录；若一条镜像地址也找不到，先停在这里核对是否确实使用本地版 `pyproject.toml`。重复执行地址替换不会把地址改坏。

### 4.3 更新锁文件来源，再安装 CUDA 依赖

**步骤 A：更新安装清单的来源。输入下面两行：**

```powershell
uv lock --refresh --python 3.12
if ($LASTEXITCODE -ne 0) { throw "更新锁文件失败，请先检查上方错误。" }
```

uv 会解析依赖，可能显示 `Resolved ... packages`。数量不必和别人完全一样。第二行检查上一条的退出码：`0` 表示成功，非 `0` 会显示停止提示。没有报错并返回提示符后，才执行步骤 B。

**步骤 B：创建环境并安装。输入：**

```powershell
uv sync --python 3.12 --extra cuda --frozen
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败，请先检查上方错误。" }
```

这一步可能先下载 Python，再创建 `.venv`、下载软件包、安装依赖。CUDA Torch 文件较大，时间取决于网络与硬盘，不代表程序卡住。不要同时再开一次相同安装。

| 命令部分 | 你需要理解的意思 |
| --- | --- |
| `uv sync` | 把这个项目的环境整理成依赖清单要求的状态 |
| `--python 3.12` | 使用 Python 3.12；缺少时 uv 可下载 |
| `--extra cuda` | 选择这个项目定义的 GPU 依赖组合 |
| `--frozen` | 使用现有锁文件，不在这一步重新计算依赖版本 |

**步骤 C：检查安装出来的文件。输入：**

```powershell
Test-Path ".\.venv\Scripts\python.exe"
.\.venv\Scripts\python.exe --version
```

预期先得到 `True`，再看到 `Python 3.12.x`。若第一项是 `False`，不要继续输入训练命令；回看步骤 B 最后面的错误。若版本号不是 3.12，检查是否用了项目内这条完整路径。

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

**本节目标：** 确认“电脑有显卡”和“这套 Python 环境能够使用显卡”两件事都成立。CUDA 是 NVIDIA 显卡提供计算能力的一套软件体系；有显卡不意味着任意安装的 Torch 都能调用它。

### 5.1 先看系统是否识别显卡

**输入：**

```powershell
nvidia-smi
```

正常会出现一个表格，包含 GPU 名称、驱动版本与显存占用。这里不需要逐项背下来，先找 `RTX 3070`。如果命令不存在，可能是驱动或命令路径问题，继续保留错误信息；第 5.2 的 Python 诊断仍有参考价值。

### 5.2 再检查项目的 Python 环境

**输入：**

```powershell
.\.venv\Scripts\python.exe -m local_rl.doctor
```

这条命令可以读成：“用本项目的 Python，运行本地训练包里的 doctor 检查程序。”它不下载模型、不训练 Qwen，只做环境检查和很小的 CUDA 反向传播。

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

### 5.3 根据结果决定下一步

| 诊断结果 | 你现在能做什么 |
| --- | --- |
| `cuda_available: true` 且 `cuda_backward: true` | 先学第 6 节，再尝试第 7 节 GPU 模型训练 |
| `cuda_available: false`，但 Python 能正常启动 | 可以先完成 CPU toy；GPU 部分按第 15 节排查 |
| Python 本身报导入错误，诊断没完成 | 回到环境安装，先不要继续训练 |

`true` 是“是”，`false` 是“否”，`null` 是“没有该值”。这些词只是 JSON 的表达方式。能读懂这三个值，就能做初步判断。

诊断成功不等于所有模型都装得下，也不证明模型已经学习。实际 Qwen 训练还要经过第 7 节两轮检查。

<a id="toy"></a>

## 6. 第一课：CPU toy

**本节目标：** 不受模型下载与显卡设置干扰，先亲眼看到一次“训练—保存—评估”。只需要安装好的 Python 环境；CPU 版和 CUDA 版 Torch 都能做这节。

`toy` 是“玩具示例”的意思，不是另一个付费软件。任务：输入两个 0–9 的整数，从 0–9 中选出最大值。例如 `(2,7)` 选 `7` 得 1 分，其他答案得 0 分。随机初始化的 1,994 参数小网络学习这个单步决策，不涉及语言建模或 LoRA。

### 6.1 先理解“训练前的它为什么会答错”

这个小网络起初没有学会比较两个数字。它会给 0–9 每个答案一个概率，再按概率选答案。假设某次输入 `(2,7)`，它仍可能选 3。

程序给正确答案 1 分、错误答案 0 分，然后调整网络内部数字，让好答案变得更可能出现。这个变化不是你手工写了 `max(a,b)` 帮它作答：`max` 只用于生成标准答案、检查它的选择。

### 6.2 运行训练命令

**前提：** 当前目录为 `E:\Agentic_RL\agentic-rl-lab`，`.venv` 已安装。`outputs/toy-01` 不应包含已有实验；如果存在，请先阅读第 15 节“重复运行”，统一换新名称。

**输入：** 下面虽然很长，仍然是一条命令，完整复制后回车。

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate 0.01 --save-every 100 --output outputs/toy-01
```

先把命令的前半段拆开看：

| 文字 | 作用 |
| --- | --- |
| `.` | 当前文件夹 |
| `.\.venv\Scripts\python.exe` | 当前项目安装的 Python |
| `-m` | 按 Python 模块名称运行程序 |
| `local_rl.train` | 本项目负责训练的入口 |
| 后面的 `--名称 值` | 给本次实验选择设置 |

这里只需要会复制并理解含义，不必从头手写这条命令。其余参数如下：

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

**运行中会看到：** 先打印可训练参数数量，再一轮一行地打印 JSON。JSON 是用“字段名：值”记录信息的格式，不需要你输入什么答案。看着 `step` 增加即可；现在不必理解每个字段。

**结束时应看到类似文字（只对照，不执行）：**

```text
Saved checkpoint: outputs/toy-01/checkpoint-000100
```

Windows 上打印的分隔符也可能是反斜杠，这不影响含义。提示符 `PS ...>` 返回后，再运行摘要：

```powershell
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/toy-01
```

先找 `rounds` 和 `optimizer_updates`。`rounds: 100` 表示尝试了 100 轮；优化次数可能小于 100，因为有些轮次没有组内差异。比如尝试 100 次、真正更新 85 次，是能够解释的情况，不是自动失败。

### 6.3 找到电脑保存的成果

**输入：** 打开这次实验的文件夹。

```powershell
explorer.exe ".\outputs\toy-01"
```

你应看到 `config.json`、`metrics.jsonl`、`rollouts.jsonl` 和 `checkpoint-000100` 文件夹。`outputs` 不需要事先创建，训练程序会创建。

路径 `outputs/toy-01` 是相对地址：在当前项目下找到 `outputs`，再找到 `toy-01`。所以每次先进入正确项目目录很重要。

### 6.4 给训练前后各做一次小测验

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/toy-01/checkpoint-000100 --base --limit 20 --output outputs/toy-01/base-eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/toy-01/checkpoint-000100 --limit 20 --output outputs/toy-01/trained-eval.json
```

两条命令**依次执行**，每条完成后再输入下一条。第一条带 `--base`，测初始网络；第二条不带，测训练后的网络。

为什么训练结束后还能测“训练前”？toy 的 `--base` 按训练时记录的随机种子重建同一个初始网络，不会抹掉训练后的文件。两次都用未参与训练的 20 道题，选择概率最大的动作。

`seed`（随机种子）可以粗略理解为给随机过程一个起点编号；它帮助复现实验，不是“填 42 就会得到正确答案”。跨软硬件仍可能有数值差异。

### 6.5 把小测验结果读成人话

完成评估后，分别输入下面两条只读命令：

```powershell
Get-Content -Encoding UTF8 -Raw ".\outputs\toy-01\base-eval.json" | ConvertFrom-Json | Select-Object count, accuracy, format_rate
```

```powershell
Get-Content -Encoding UTF8 -Raw ".\outputs\toy-01\trained-eval.json" | ConvertFrom-Json | Select-Object count, accuracy, format_rate
```

这一长行只是“读文件 → 解析 JSON → 只显示三列”，不修改结果。竖线 `|` 把前一步读到的内容交给下一步处理。

| 显示的值 | 读成什么 |
| --- | --- |
| `count = 20` | 测了 20 道题 |
| `accuracy = 0.1` | 正确率 10%，即 20 题里 2 题正确 |
| `accuracy = 0.55` | 正确率 55%，即 20 题里 11 题正确 |
| `format_rate = 1.0` | 100% 回答符合本程序的答案格式 |

最后一项和正确率不是同一个概念：格式全部正确，也可能有很多题答错。toy 程序会把选择的数字包装成合法文本，因此这里格式率为 1 很正常。

固定 seed 验证曾得到初始 `2/20`、训练后 `11/20`。这是一次小任务观察，不是你必须达到的分数，也不是语言模型能力证据。

**完成标志：** 有训练日志、检查点、实际优化次数和两份独立评估 JSON。你还能回答：初始网络答对多少题？训练后答对多少题？运行了多少轮、真正更新了多少次？

到这里就可以结束第二次学习。即使暂时不下载 Qwen，你已经跑过真实的参数更新流程。下一次开终端只需回到项目目录，成果仍在 `outputs/toy-01`，不用重新安装。

<a id="model"></a>

## 7. 第二课：下载模型与 GPU 训练检查

**本节目标：** 准备一个真正会生成文字的小模型，并确认电脑能够修改它的 LoRA 参数。

**开始条件：** 第 5 节 CUDA 检查通过；第 6 节至少成功完成一次训练与评估。没有 GPU 条件时先停留在 toy，不必为学习第一课购买算力。

### 7.0 先把三个新词讲清楚

**Qwen2.5-0.5B-Instruct** 是模型名称。`0.5B` 表示约 5 亿参数规模；`Instruct` 表示模型经过指令相关训练，能够接收问题生成回答。它与前面只有 1,994 个参数的 toy 不是同一个模型。

**LoRA** 是一种少量参数微调方法。可以先理解为：保留原来的大本书，在旁边训练一小本补充笔记；回答时两者共同起作用。检查点主要保存这本“小笔记”，所以加载时还需要原模型。这个比喻只说明保存与更新范围，LoRA 实际上是附加的低秩矩阵。

**token** 是模型处理文字的小单位，不等于固定一个汉字或一个单词。把文字拆成这些单位的工具叫 tokenizer（分词器）。`max-new-tokens=96` 是限制新生成单位数，不是要求答案正好写 96 个字。

### 7.1 下载与训练是两件事

默认模型 `Qwen/Qwen2.5-0.5B-Instruct` 的公开权重可以匿名下载，不需要购买在线推理或训练服务。为区分网络故障与训练故障，先单独下载到 Hugging Face 缓存：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', max_workers=2))"
```

这条命令中的 `-c` 表示“执行引号中的这段 Python”，不同于前面的 `-m`。`snapshot_download` 负责下载模型文件，`max_workers=2` 表示最多用两个并行下载工作线程；**2 不是训练轮数**。

**运行时：** 可能依次显示配置、tokenizer、权重等文件的下载进度。某一个文件显示 100% 不一定代表整个模型都完成。

**完成标志：** 返回一个本地 snapshot 路径，并重新出现 PowerShell 提示符，没有下载异常。默认缓存通常在用户目录下，因此项目放在 E 盘，模型缓存仍可能在 C 盘。

这是下载成功，还没有训练。缓存目录请保留；LoRA 检查点不会另外打包整个基础模型。中断下载后通常可以重试同一条命令，工具会复用已完成的文件，不要急着清空缓存。

PyPI 镜像不负责模型下载。若 `huggingface.co` 连接失败，可选用 [HF-Mirror](https://hf-mirror.com/) 的第三方下载镜像，在**当前 PowerShell**设置后重试上面的命令：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
$env:HF_HUB_DISABLE_XET = "1"
```

三行分别表示：改用镜像端点、不自动附带已有账号 token、禁用 Xet 下载方式以使用另一条下载路径。执行设置本身不会下载文件；**接着重新运行本节最前面的下载命令**才开始下载。

它不是 Hugging Face 官方站点，本教程仅下载公开文件，不传账号 token。镜像也可能临时不可用。恢复默认端点可执行 `Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue`。完整缓存后，训练加 `--offline`。

`--offline` 的含义是“只找本地已有文件，不尝试上网补齐”。它不会凭空生成缺少的模型，也不能让尚未下载的文件直接可用。

也可以把完整模型放到自选目录，使用 `--model "E:/Models/Qwen2.5-0.5B-Instruct" --offline`。目录要有配置、tokenizer 和真实权重；只有一个权重文件或 Git LFS 指针不够。**改用自定义路径后，后续训练、加载适配器和续训都保持同一个 `--model` 值。** 下文统一采用默认模型名和缓存。

### 7.2 两轮 SFT：验证真实 GPU 反向传播

SFT 可以先理解为“看标准答案学习”：程序给出一道加法题，同时提供正确回答，模型根据回答误差调整可训练参数。反向传播就是计算“该怎样调整”；优化器负责真正执行调整。

**输入：** 保持项目根目录，完整复制这一行。此时模型必须已经完整缓存。

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --objective sft --offline --steps 2 --max-new-tokens 64 --save-every 2 --output outputs/gpu-check-01
```

SFT 直接使用本地算术标准答案，检查前向、反向、优化与保存；不依赖随机回答是否有对有错。核对 `optimizer_updates` 增加、`grad_norm` 有限且有学习信号，并生成 `checkpoint-000002/manifest.json`。

用一个表把完成条件对应到具体动作：

| 你要确认的事 | 怎么确认 |
| --- | --- |
| 程序确实开始了训练 | 出现 `Trainable parameters: 540,672` 及 `step` 日志 |
| 完成了两轮 | 日志中的 `step` 到 2，并返回提示符 |
| 做过参数更新 | 正常完成两轮 SFT 时，累计 `optimizer_updates` 应到 2 |
| 保存完整 | 下面的文件检查返回 `True` |

```powershell
Test-Path ".\outputs\gpu-check-01\checkpoint-000002\manifest.json"
```

若更新次数没增加或出现异常，先保留完整日志，不要继续批量启动其他实验。正常 SFT 的 `reward` 仍可能显示 0，因为这里没有用随机生成回答打奖励分；第 10 节还会解释。

这是运行检查，不证明两轮就让模型变强。540,672 是需要更新的 LoRA 参数数量，不是把约 5 亿参数的模型删除成 54 万参数。

精度、梯度缩放、梯度检查点等设置已由代码处理，第一次不要更改。具体默认是 CUDA 上优先 BF16，否则 FP16 + 梯度缩放；CPU 为 FP32。无需为完成本节额外安装 FlashAttention、DeepSpeed 或 vLLM。

<a id="grpo"></a>

## 8. 第三课：真正运行 GRPO

**本节目标：** 让模型自己尝试多种回答，用“同一道题里的相对好坏”决定怎样更新。

SFT 类似看参考解答；GRPO 类似自己做几遍题，程序只给每次尝试打分，再比较哪次更好。两种方式都有训练价值，只是学习信号不同。

先读一下名称，不必背缩写：GRPO 的 G 对应 group（组），R 对应 relative（相对）。这里最关键的是“**同一道题的一组回答，相互比较**”。

### 8.1 从本地算术题开始

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --steps 20 --batch-size 1 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo-01
```

默认 `--backend hf --task arithmetic --objective grpo`：真实 Qwen、内置加法题、GRPO。`hf` 表示使用 Transformers 这一套本地模型实现，不表示调用收费的 Hugging Face 推理服务。

第 7 节的 SFT 检查点**不会自动加载**；这里从原始 Qwen 加新 LoRA 开始。先前模型文件仍在缓存中，不必再下载一次。

`outputs/qwen-grpo-01` 是本次实验的名称，你可以理解为新建一本实验记录本。第一次原样使用；重新实验时可以叫 `qwen-grpo-02`，但后面查看、评估的路径也要一起换。

**如果你先前已经跑过 `outputs/grpo-20`：** 可以直接学习第 9 节如何读现有日志，不必为了名字相同再训练一次。把阅读和评估命令中的实验路径替换为自己的实际路径即可。续训则不仅要改路径，还要保留原始设置；例如先前 `max-new-tokens=64`，就不能直接照抄本节 96 的续训参数。

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

几个容易误会的数字：

- `batch-size=1, group-size=4` 是“一道题，四个回答”，不是“四道不同题”。
- `samples=64` 是抽题范围有 64 道，不是每轮生成 64 个回答。
- 每轮抽题可能重复，因此 20 轮不保证做过 20 道不同题。
- `2e-5` 是科学计数法，等于 `0.00002`；先保持默认学习率。
- 96 是最多生成的 token 预算，模型正常结束时可以远少于 96。

题库是两个 0–19 整数相加，不下载 GSM8K。400 道题固定划为 320 训练题与 80 评估题；`--samples 64` 只缩小训练池，不改变评估集。

### 8.2 一轮发生了什么

1. 从训练池随机抽题，用当前策略生成四个回答。
2. 本地规则给每个回答打 0 或 1 分。
3. 比较组内奖励，计算优势。
4. 保存旧策略对已采样 token 的 log probability，再前向计算可求导的新概率。
5. 算策略损失，逐条 backward，更新 LoRA。
6. 写日志，按间隔保存检查点。

把“概率”先理解成模型有多倾向于输出某个 token。log probability 是概率的一种数学表示，便于稳定计算；第一次只需要知道程序用它比较“更新前后，模型对同一个回答的倾向变了多少”。

模型、策略、policy 在这些日志中大多指产生回答的那套规则。rollout 指模型实际尝试生成出来的一次回答或工具交互过程。它们都不是需要另外安装的软件。

生成和反向传播逐条执行，避免同时保留整批生成缓存或计算图；没有同时加载四份模型，也没有远程求导。

刚启动时要加载模型，所以第一条 step 日志不一定立即出现。进入采样后，通常一整轮完成才打印一行，生成期间不会把每个 token 都实时显示在终端。不要因为几秒没有新文字就重复启动第二个训练进程。

**完成标志：** 保存 `checkpoint-000020`，能区分“20 轮采样”和“多少次有效更新”。零更新时先检查奖励与文本，不能把它当成已学会任务。

<a id="logs"></a>

## 9. 奖励、优势、损失与日志

**本节目标：** 读懂自己刚才运行的程序，而不只知道“命令没报错”。先讲判分，再手算一组优势，最后对照真实日志。

### 9.1 奖励完全在本地计算

对语言模型，最终答案须有 `\boxed{数字}`。程序读取最后一个框，解析数值，与标准答案比较。

| 标准答案为 7 时的文本 | 奖励 | 原因 |
| --- | --- | --- |
| `The answer is \boxed{7}.` | 1 | 格式与数值正确 |
| `\boxed{7.0}` | 1 | 数值等价 |
| `The answer is 7.` | 0 | 没有要求的答案框 |
| `\boxed{8}` | 0 | 数值错误 |
| `\boxed{3+4}` | 0 | 最终答案不接受表达式 |

`\boxed{7}` 在这里首先是一段需要出现的文本，程序从中提取数字。你不需要安装 LaTeX，也不需要手工给每条模型回答加框；应观察模型是否自己遵守。

为什么数字对了还可能 0 分？当前奖励同时检查“格式”和“数值”。这是明确定义的任务规则，不是自动理解一切表达的智能裁判。训练失败、格式不合规、数学答错，是三件不同的事。

不使用付费裁判模型。开放式文章和主观解释不能直接用这个数值奖励。

### 9.2 手算组内优势

假设问题是“3 + 4 等于多少”，四个候选是下面这样。这是**手算用的教学示例，不是你必然生成的文本**：

| 候选 | 回答 | 奖励 | 相对评价 |
| --- | --- | --- | --- |
| A | `\boxed{6}` | 0 | 比这一组平均水平差 |
| B | `The answer is 7.` | 0 | 缺少格式，按规则不得分 |
| C | `\boxed{7}` | 1 | 比这一组平均水平好 |
| D | `7, so \boxed{7}` | 1 | 比这一组平均水平好 |

逐步计算：

1. 求平均分：`(0+0+1+1)/4 = 0.5`。
2. 每个分数减平均：`[-0.5,-0.5,+0.5,+0.5]`。
3. 这些差值的平方都是 `0.25`，平均后仍是 `0.25`。
4. 对 `0.25` 开平方得到总体标准差 `0.5`。你暂时只需把它理解为衡量这一组分数差异的量。
5. 第 2 步的差值再除以 `0.5`，得到 `[-1,-1,+1,+1]`，这就是默认的 advantage（优势）。

压缩成公式是：

```text
advantage = (reward - 组内均值) / max(组内总体标准差, 0.000001)
          = [-1, -1, +1, +1]
```

正优势倾向提高该回答动作的概率，负优势倾向降低。全错 `[0,0,0,0]` 或全对 `[1,1,1,1]` 的优势都为零，没有组内相对信号。

为什么全错时不能继续按同样方法“奖励相对好的回答”？因为它们的分数全一样，规则没有告诉模型哪条更好。即使一条写得长、一条写得短，只要本次规则都判 0，就没有这种相对信号。

全对也是同理：这个组没有给出谁比谁更好的信息，不代表模型坏了。程序会把这种组的优势设为零，避免除以零。

默认 `beta=0` 时，整轮没有有效组就跳过优化器更新。`beta` 与额外 KL 约束有关，第一次保持 0 即可。`--no-normalize-advantages` 是进阶对照选项，只减均值、不除标准差，初学时无需改。

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

三条命令依次使用即可：第一条给汇总，第二条看前 4 个回答，第三条看最后 3 轮指标。都只读文件，不会再训练。

JSON 是用字段名与值保存数据的文字格式，JSONL 是每行一个 JSON。默认 batch=1、group=4 时，前四条是第一组候选。JSON 字符串会转义反斜杠，`\\boxed` 表示实际文本中的 `\boxed`，不是模型真的多打了一个反斜杠。

如果一行 JSON 太挤，下面把第一条回答展开显示；先确认第 8 节产生了这个文件：

```powershell
Get-Content -Encoding UTF8 ".\outputs\qwen-grpo-01\rollouts.jsonl" -TotalCount 1 | ConvertFrom-Json | Format-List question, answer, text, reward, advantage, tokens, truncated
```

读的时候只问三个问题：`question` 问什么？`text` 真正答了什么？对照 `answer` 和格式规则，`reward` 为什么是这个值？

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

### 9.4 逐项读一轮真实日志

下面取自读者提供的 Windows CUDA 日志片段，**只展示了其中一轮**。对应的回答预算是 64 tokens，不能直接当成本章 96 tokens 配置的性能保证。数字经过少量小数位简化：

```json
{
  "step": 1,
  "optimizer_updates": 1,
  "loss": 0.0,
  "reward": 0.5,
  "active_groups": 1,
  "attempted_groups": 1,
  "grad_norm": 6.6192,
  "mean_tokens": 22.0,
  "truncated_fraction": 0.25,
  "seconds": 9.8402,
  "peak_allocated_mib": 1089.69,
  "peak_reserved_mib": 1192.0
}
```

这是**示例输出，不放进 PowerShell 执行**。将它翻译成人话：

| 日志片段 | 本轮能得出的结论 |
| --- | --- |
| step 1、updates 1 | 完成第一轮，并且累计做过一次优化器更新 |
| reward 0.5 | 本轮一个组共四个回答，按当前 0/1 规则有两个得分 |
| active_groups 1 | 这一组存在组内奖励差异，可以产生策略梯度信号 |
| loss 0、grad_norm 6.6192 | 损失数值是零，但梯度不是零；不能说“没训练” |
| truncated_fraction 0.25 | 四条轨迹中一条被标记为未正常结束，原因还要看原文 |
| mean_tokens 22 | 每条平均生成 22 个动作 token，不是每条恰好 22 个字 |
| seconds 9.8402 | 这轮采样与计算约 9.84 秒，不含初始下载与加载 |
| allocated 1089.69 MiB | 这轮 PyTorch 活跃分配峰值约 1.06 GiB，不是整卡总占用 |

`grad_norm` 显示的是**裁剪前**数值，所以大于 1 不等于梯度裁剪失效。程序实际更新前会把梯度范数限制到 1。

下一轮如果 `step` 到 2，但 `optimizer_updates` 仍是 1，且 `reward=0`、`active_groups=0`，意思是“又尝试了一轮，这轮全错没有相对信号，累计更新次数没变”。这与程序崩溃不同。

只有拿到最后一轮记录、完整检查点和评估文件，才能判断整个实验是否完成。不能从这一行断言整次实验只占 1GB 显存，或已经提升了数学能力。

### 9.5 三个零，含义不同

| 看见的零 | 可能含义 | 应该再看什么 |
| --- | --- | --- |
| `reward=0` | 此轮回答按规则都没得分 | 原始回答、答案格式、截断 |
| `active_groups=0` | 没有组内相对信号 | 是否全错或全对 |
| `loss=0` | 可能是正负项数值抵消，也可能根本没更新 | 梯度与累计更新次数 |

损失是程序用来求导的目标值，奖励是回答得分。它们不是同一指标，不能用“loss 越接近 0，模型就一定越好”来判断本实验。

### 9.6 可选画本地曲线

```powershell
uv sync --python 3.12 --extra cuda --extra plots --frozen
if ($LASTEXITCODE -ne 0) { throw "绘图依赖安装失败。" }
.\.venv\Scripts\python.exe -m local_rl.analyze outputs/qwen-grpo-01 --plot
```

图在 `outputs/qwen-grpo-01/learning.png`，无需 SwanLab。以后同步依赖时保留所需 extra；只同步 `cuda` 可能移除先前可选的绘图依赖。

<a id="sft"></a>

## 10. 没有学习信号时先做 SFT

**本节目标：** 当模型还不会按规则答题时，用参考答案帮助它学习，再检查是否适合进入 GRPO。不是看到一轮 0 分就必须重训，也不是所有人都要反复做这一步。

小模型可能会加法，却不遵守答案格式，或在 96 tokens 内没给最终答案。经常全零奖励时先读 rollout，再决定是否 SFT。

SFT（监督微调）直接展示“题目 → 标准回答”，优化回答 token 的交叉熵；参考答案来自本地算术程序，不购买教师调用。

### 10.1 先训练一个 SFT 适配器

**输入：**

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --objective sft --steps 30 --output outputs/qwen-sft-01
```

`--objective sft` 把学习目标切换为监督微调；输出目录另起名称，避免覆盖前面的 GRPO。它从基础模型加新适配器开始，不会自动继承刚才直接 GRPO 的参数。

预期正常结束后得到 `checkpoint-000030`。不要求每轮 loss 都比上一轮小，因为每轮抽到的题可能不同。

### 10.2 先看它是否更会按格式回答

**训练结束后输入：**

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-sft-01/checkpoint-000030 --offline --limit 20 --output outputs/qwen-sft-01/eval.json
```

查看摘要：

```powershell
Get-Content -Encoding UTF8 -Raw ".\outputs\qwen-sft-01\eval.json" | ConvertFrom-Json | Select-Object count, accuracy, format_rate, truncated_fraction
```

如果格式率仍为 0，先看这份 JSON 的 `results` 里实际写了什么，不必立即启动下一阶段。可能需要更多 SFT、调整回答预算，或先简化任务。30 轮不是保证达标的次数。

### 10.3 再从这个适配器开始 GRPO

明确告诉程序“加载刚刚练习过的 LoRA 笔记”，开始新实验：

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --adapter outputs/qwen-sft-01/checkpoint-000030 --steps 20 --group-size 4 --max-new-tokens 96 --output outputs/qwen-sft-grpo-01
```

`--adapter` 后面填的是完整检查点文件夹，不是 `outputs/qwen-sft-01` 这一层，也不是某个 `.safetensors` 文件。它会加载 LoRA 权重，再开始新的优化过程。

SFT 日志的 `reward=0` 是占位，**不能据此判断没有学习**；看 loss、梯度和刚刚保存的生成评估。进入 GRPO 后，reward 才是这次采样回答按规则得到的分数。

SFT 用参考答案教格式与解答，GRPO 比较自己生成的候选；两者都更新参数，学习信号不同。

<a id="evaluation"></a>

## 11. 独立评估

**本节目标：** 回答“模型在未参与训练的题目上表现如何”，而不只回答“程序有没有跑完”。

训练题像练习册，评估题像另外一份小测验。如果拿刚练过的原题考试，可能只是记住了这些题，无法说明对新题的表现。本项目提前固定了训练与评估题目，默认评估不会修改模型参数。

### 11.1 基模与直接 GRPO 对照

完成第 8 节后运行：

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo-01/checkpoint-000020 --base --offline --limit 20 --output outputs/qwen-grpo-01/base-eval.json
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-grpo-01/checkpoint-000020 --offline --limit 20 --output outputs/qwen-grpo-01/trained-eval.json
```

**执行顺序：** 先等带 `--base` 的第一条结束，再执行第二条。两份输出路径必须不同；不要为了重试删掉原结果，换一个新文件名即可。

| 参数 | 意思 |
| --- | --- |
| `--checkpoint ...checkpoint-000020` | 读取这次实验保存的模型 / 配置 |
| `--base` | 本次跳过已训练的适配器，测试原始基模 |
| `--limit 20` | 最多检查 20 道保留题 |
| `--output ...json` | 将这次测验的结果保存到指定文件 |
| `--offline` | 只使用本地已有模型文件 |

为什么测“训练前”也要给训练后的检查点路径？它用这份记录确定同一个模型、任务、长度等设置，再跳过训练后的适配器。原始基模权重一直保留着，不会因评估被改写。

两次使用相同评估题与长度设置，采用 greedy（贪心）解码：每一步选概率最高的 token。训练则为了得到不同候选会进行随机采样。因此训练奖励与评估分数不要求相等。

| 字段 | 含义 |
| --- | --- |
| `count` | 实际评估题数 |
| `accuracy` | 合法答案框且数值正确的比例 |
| `format_rate` | 能解析合法数值答案框的比例，不要求数值正确 |
| `truncated_fraction` | 被标记为未正常结束的比例 |
| `results` | 每题参考答案、生成原文、正确性与工具观察 |

### 11.2 把两份分数放在一起看

依次输入：

```powershell
Get-Content -Encoding UTF8 -Raw ".\outputs\qwen-grpo-01\base-eval.json" | ConvertFrom-Json | Select-Object count, accuracy, format_rate, truncated_fraction
```

```powershell
Get-Content -Encoding UTF8 -Raw ".\outputs\qwen-grpo-01\trained-eval.json" | ConvertFrom-Json | Select-Object count, accuracy, format_rate, truncated_fraction
```

先比较 `count` 是否相同，再比较其他指标。例如正确率由 0.2 变成 0.3，就是 20 题从答对 4 题变成 6 题；这是**说明怎样读数的例子，不是实际效果承诺**。

| 观察到的结果 | 能怎样解释 |
| --- | --- |
| 格式率上升，正确率没变 | 更遵守输出格式，但不能说数学准确性已经提高 |
| 训练奖励升高，保留题分数没升 | 训练与评估分布 / 解码不同，也可能没有泛化提升 |
| 两次都为 0 | 先看文本：格式、截断、计算错误各占多少 |
| 小样本分数上升 | 记录观察，再用更多保留题或更多种子检查稳定性 |

查看一题原文时可输入：

```powershell
(Get-Content -Encoding UTF8 -Raw ".\outputs\qwen-grpo-01\trained-eval.json" | ConvertFrom-Json).results | Select-Object -First 1 | Format-List question, answer, text, correct, formatted, truncated
```

`correct` 表示是否按完整规则答对；`formatted` 表示是否有合法数值答案框。不要只截一张“奖励上涨”的图就结束分析。

### 11.3 做过 SFT 时比较三种状态

第 10.2 节已经保存了 SFT 的评估文件，不需要向同一个文件再评估一次。完成第 10.3 节后，只增加 SFT 后 GRPO 的这份评估：

```powershell
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/qwen-sft-grpo-01/checkpoint-000020 --offline --limit 20 --output outputs/qwen-sft-grpo-01/eval.json
```

| 模型状态 | 结果文件 |
| --- | --- |
| 原始基模 | `outputs/qwen-grpo-01/base-eval.json` |
| SFT 后 | `outputs/qwen-sft-01/eval.json` |
| SFT 后继续 GRPO | `outputs/qwen-sft-grpo-01/eval.json` |

先确认三个实验使用相同模型与评估设置，再比较。

加上上一节基模结果，就有“基模 → SFT → SFT 后 GRPO”三个观测点。若改过模型、数据或生成预算，应统一条件后重测，不能混比。

`--base` 永远指**原始基模**，不自动指 GRPO 起点的 SFT 适配器；评估 SFT 起点必须单独指定其检查点。

训练与评估题目固定分割且不相交。20 题只是小样本检查，可设 `--limit 80` 评估全部算术保留题；题库不足时报告实际数量。反复用保留集调参后，它成为验证集，正式结论还需要额外未参与调参的数据。

真实 Qwen 的短程 CPU 检查只验证流程贯通：一次 4 题评估中，训练前后严格 boxed 正确率都为 `0/4`。**没有据此声称真实模型性能提升。** 详见[验证记录](../../docs/local/validation.md)。

<a id="checkpoints"></a>

## 12. 保存、续训与加载适配器

**选读：** 第一次只想跑通训练与评估，可以暂时跳过。准备延长训练或处理中断时再读。

检查点（checkpoint）像学习过程的存档。日志记录“发生过什么”，检查点保存“模型和优化器现在是什么状态”，两者不能互相替代。

默认每 10 轮及最后一轮保存。`manifest.json` 最后写入，表示保存完成；中断时选择最后一个包含它的检查点。

| 内容 | 用途 |
| --- | --- |
| LoRA 与 tokenizer 文件 | 加载训练后策略，仍需原基模 |
| `training.pt` | 优化器、混合精度缩放与随机状态 |
| `manifest.json` | 参数、轮次、更新次数、数据指纹、依赖版本和可获得的模型 revision |

toy 保存完整小网络；Qwen 保存 LoRA，不是整个 0.5B 模型。

### 12.1 先确认存档存在

下面以**原样执行第 8 节**为前提。先输入：

```powershell
Test-Path ".\outputs\qwen-grpo-01\checkpoint-000020\manifest.json"
```

结果必须是 `True`。目录名里的 `000020` 是第 20 轮编号，前面的零只是让排序整齐；它不保证已做了 20 次优化更新。

如果你中途在第 13 轮停止，可能只有 `checkpoint-000010`，这时不能凭空填写 `000020`。先找到实际完整保存的那个目录。

### 12.2 继续同一次实验

将第 8 节的 20 轮实验继续到总共 40 轮，**输入：**

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --resume outputs/qwen-grpo-01/checkpoint-000020 --steps 40 --batch-size 1 --group-size 4 --max-new-tokens 96 --output outputs/qwen-grpo-continued-01
```

这会执行第 21–40 轮，使用新输出目录，最终生成 `checkpoint-000040`。`--steps 40` 不是“再做 40 轮”。没有保存的轮次需要重跑。

没有指定 `--resume` 时，即使用一样的参数和 seed，程序也会开始新实验，不会自动在旧进度上继续。

### 12.3 继续旧实验和借用旧权重是两回事

| 选项 | 恢复什么 | 用途 |
| --- | --- | --- |
| `--resume` | 权重、优化器、随机状态、累计更新次数 | 继续同一实验 |
| `--adapter` | 适配器权重，重建训练状态 | SFT 后 GRPO、开始新实验 |

续训不会自动补齐所有 CLI 参数，应保持原算法、学习率、模型路径、seed、数据、batch / group 等关键配置；具体以 `config.json` 为准。想换设置就开始新实验，不要修改 manifest 绕过检查。

**当前限制：** 从 `--adapter` 热启动的实验不能直接按现有 CLI 精确续训，因为配置校验要求保留该参数，而 CLI 又禁止它与 `--resume` 同时出现。此时可用最新检查点的 `--adapter` 开始新实验，但会重置优化器 / 随机状态，不应称为精确恢复。上面从基模开始的第 8 节示例不受此限制。

CPU 测试覆盖同配置续训与不中断训练逐参数一致；跨硬件、驱动或精度不保证数值相同。

<a id="extensions"></a>

## 13. 其他算法、工具任务与 OPD

**进阶选读。** 下表是后续课程索引，第一次不需要理解所有缩写。先完成 toy、GPU 检查与一次独立评估，再选择一项尝试，不同时启动多个任务。

想知道“换一种算法会怎样”，再读 13.1；想理解“模型如何使用计算器”，从 13.2 开始；想探索“一个本地模型怎样给另一个提供学习信号”，再读 13.3。

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

**进阶选读。** 内置算术任务无需准备数据文件。只有想换成自己的题目时才做这一节，不要把“必须先下载数据集”当成入门条件。

### 14.1 离线 JSONL

每行必须含 `question`、数值 `answer` 和 `split`：

```json
{"question":"What is 1 + 2?","answer":"3","split":"train"}
{"question":"What is 7 + 8?","answer":"15","split":"eval"}
```

逐项理解：`question` 是题目，`answer` 是标准数值答案，`split` 指定这条题用于训练还是评估。`train` 是练习题，`eval` 是测验题；字段名称与英文取值保持原样。

上面只是格式示例。下面提供一份更方便创建的微型文件，仍只用于验证数据读取，不是效果评测数据集。

**在项目根目录整块执行：** 它创建 `data` 文件夹和 `math-beginner.jsonl`。如果同名文件已存在，整块代码会停止，不覆盖。外层 `& { ... }` 也一并复制。

```powershell
& {
$mathDataPath = Join-Path (Get-Location) "data\math-beginner.jsonl"
if (Test-Path $mathDataPath) {
    throw "数据文件已存在，请先检查原文件；这一步不需要重复执行。"
}
New-Item -ItemType Directory -Path ".\data" -Force | Out-Null
$mathRows = @'
{"question":"What is 1 + 2?","answer":"3","split":"train"}
{"question":"What is 2 + 4?","answer":"6","split":"train"}
{"question":"What is 3 + 5?","answer":"8","split":"train"}
{"question":"What is 4 + 6?","answer":"10","split":"train"}
{"question":"What is 7 + 8?","answer":"15","split":"eval"}
{"question":"What is 8 + 9?","answer":"17","split":"eval"}
{"question":"What is 9 + 10?","answer":"19","split":"eval"}
{"question":"What is 10 + 11?","answer":"21","split":"eval"}
'@
[System.IO.File]::WriteAllText($mathDataPath, $mathRows, [System.Text.UTF8Encoding]::new($false))
Get-Content -Encoding UTF8 $mathDataPath
}
```

`@'` 和 `'@` 把中间多行文字作为一整段数据；请完整复制，结束标记单独一行。`UTF-8` 是文字编码，`无 BOM` 是本程序需要的文件开头格式，以上命令已替你处理。

**预期结果：** 打印 8 行题目，前 4 行 train，后 4 行 eval。文件名必须是 `.jsonl`，不是 `.jsonl.txt`。下一条训练命令会明确指向这份文件，不会自动读取聊天内容。

```powershell
.\.venv\Scripts\python.exe -m local_rl.train --device cuda --offline --task jsonl --data data/math-beginner.jsonl --steps 20 --output outputs/custom-01
.\.venv\Scripts\python.exe -m local_rl.evaluate --checkpoint outputs/custom-01/checkpoint-000020 --offline --output outputs/custom-01/eval.json
```

这份小文件只有 4 道评估题，所以评估 `count` 应为 4，即使默认上限是 20。`--samples 64` 也不会把 4 道训练题自动扩充成 64 道，程序只使用现有题目。

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

遇到问题时，先停在当前步骤，不连续更换十几个参数。先判断失败发生在哪一层，会更容易解决。

| 失败发生的位置 | 优先检查什么 |
| --- | --- |
| `Set-Location` 就失败 | 文件夹是否存在、有没有多套一层 |
| `uv sync` 失败 | 下载地址、网络、版本、磁盘空间 |
| Python 无法启动或 import 失败 | 当前 `.venv` 是否安装完整 |
| 下载模型失败 | HF 连接与缓存，不是 GRPO 算法 |
| 模型加载后 GPU 报错 | CUDA 环境、显存和实际模型设置 |
| 训练正常结束但分数低 | 数据、奖励规则、回答格式、长度、学习信号 |

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
| `Sliding Window Attention ... sdpa` 警告 | 先检查后续是否仍正常打印更新日志；见下面说明，不为消除文字而盲装新组件 |
| 输出目录 / 评估文件已存在 | 将 `...-01` 改为 `...-02`，保护已有结果 |
| `Resume config mismatch` | 对照原 config 保持参数；改变设置时新建实验 |
| JSONL BOM / 解码错误 | 保存 UTF-8 无 BOM；留意 Windows PowerShell 5 默认编码 |
| loss / 梯度非有限 | 保留失败日志，检查学习率、精度和改动，恢复默认小模型配置 |

关于 `Sliding Window Attention`：当前锁定的 Transformers 版本中，警告触发检查与模型的启用开关并不完全一致。已核对的默认 Qwen 配置为 `use_sliding_window=false`、窗口值 32768，而本教程总长度上限是 512。出现这句警告本身不能说明 GPU 没工作；先看训练是否继续、更新次数和完整错误。换过模型或上下文设置后不能直接套用这个判断。先保持验证过的依赖组合，不必为了隐藏警告盲目安装 FlashAttention 或升级所有库。

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

**临时停止：** 在正在训练的 PowerShell 中按一次 `Ctrl+C`。这会中断当前程序，不保证临时替你保存最新一轮。先等进程结束、提示符返回，再做其他操作。

**以后继续：** 选最后一个有 `manifest.json` 的完整检查点，按第 12 节操作；没保存的轮次会丢失。关闭窗口后，已保存到硬盘的日志与检查点仍保留，显存占用则应随进程结束释放。

**重新做一遍：** 训练拒绝向非空输出目录重跑，评估也拒绝覆盖已有文件。这是在保护你之前的实验，不是需要重装环境的错误。

例如第一次叫 `outputs/toy-01`，下一次可以叫 `outputs/toy-02`。之后摘要、检查点路径和评估输出都改用 `toy-02`，不要混着读上一遍的结果。不要只改训练命令，却仍在旧目录看日志。

### 15.5 怎样提供有用的报错信息

遇到无法判断的问题时，保留以下内容，别人才能知道问题发生在哪里：

1. 你执行的**完整命令**，包括输出目录和所有选项。
2. 当前目录，可以用 `Get-Location` 查看。
3. 错误最后约 30–50 行，包含 `Traceback` 和最末尾的具体错误名称；不要只发“出错了”。
4. GPU 相关问题再附 `local_rl.doctor` 输出；如果诊断本身就失败，附它的报错即可。

这些本地命令不需要 API Key，日志里也不应为了求助主动加入任何账号 token。

<a id="learning"></a>

## 16. 读代码、练习与完成标准

### 16.1 不会 Python 时，先学会“观察”，再开始修改

不需要一上来从头读懂几百行代码。先打开 `local_rl/data.py`，在编辑器中用 `Ctrl+F` 搜索 `def reward`。`def` 表示定义一个函数；这里的函数接收模型文本和标准答案，返回分数。

再搜索 `if task == "toy"`，观察题库怎样生成。先只读，不要立刻改奖励。你可以把“题目、回答、得分”与第 9 节的实际日志对应起来，再学习 Python 的函数、条件判断、列表和循环。

如果还没有代码编辑器，文件可以用记事本查看；想修改时再选择熟悉的编辑器，不必为了第一次运行先装一大套插件。`.py` 是代码，`.json` / `.jsonl` 是数据或记录，`.safetensors` / `.pt` 是二进制模型或状态文件，不按普通文本阅读。

### 16.2 按执行顺序读代码

| 文件 | 要回答的问题 |
| --- | --- |
| [`local_rl/data.py`](../../local_rl/data.py) | 怎样分训练 / 评估？什么答案得分？ |
| [`local_rl/policy.py`](../../local_rl/policy.py) | 怎样生成、哪些参数更新、token 如何对齐？ |
| [`local_rl/losses.py`](../../local_rl/losses.py) | 优势、概率比、裁剪与 KL 怎么算？ |
| [`local_rl/train.py`](../../local_rl/train.py) | 何时采样、backward、step、跳过更新？ |
| [`local_rl/checkpoint.py`](../../local_rl/checkpoint.py) | 如何保存随机状态、优化器与数据指纹？ |
| [`local_rl/evaluate.py`](../../local_rl/evaluate.py) | 如何公平比较基模与适配器？ |

逐段解释见[从采样到更新](../../docs/local/walkthrough.md)。先思考：`generate()` 为什么不保留计算图，训练时却要再前向一次？

### 16.3 五个动手练习

1. 从 toy 取一组奖励，手算均值、总体标准差和优势，对照日志。
2. 固定其他条件，比较 group-size 2 与 4 的有效组比例和耗时。
3. 找到 `loss≈0` 但 `grad_norm>0` 的记录，解释标量与导数的区别。
4. 比较基模、SFT、SFT 后 GRPO 的格式率和正确率，保留原文证据。
5. 查看工具轨迹，区分模型动作与工具观察，确认观察不是策略目标。

每次只改一个因素，使用新目录。记录模型 / revision、seed、样本数、长度预算、轮次、实际更新次数与评估条件。

### 16.4 给自己留一页实验记录

可以在记事本里记录，不要求使用任何在线平台：

| 项目 | 你要填写什么 |
| --- | --- |
| 实验名称 | 例如 `toy-01` 或 `qwen-grpo-01` |
| 完整命令 | 原样保存，之后才能核对和续训 |
| 总轮次 / 实际更新次数 | 从摘要读取，分开写 |
| 初始评估 / 训练后评估 | 写题数与答对题数，不只写百分比 |
| 主要问题 | 格式不合规、截断、全错组、显存等 |
| 下一次只改哪一项 | 例如只改 group-size，保持其他条件相同 |

不要用“结果不错”代替具体记录。哪怕分数没涨，能指出原因也有学习价值。

### 16.5 三个自测问题

先自己想，再看后面的参考答案：

1. 第 5 轮结束，`optimizer_updates=3`，是不是丢了两轮？
2. `format_rate=1`、`accuracy=0.25`，是不是所有题都做对？
3. 只把 LoRA 目录复制到一台没有基础模型的电脑上，能直接完整离线推理吗？

参考答案：第 1 题，不一定，有些轮次可能因没有奖励差异跳过更新；第 2 题，不是，所有答案格式合规但只有四分之一答对；第 3 题，不能，LoRA 需要配套的基础模型与 tokenizer。

### 16.6 本章完成标准

- 能解释本机和模型下载站的职责，无需付费训练服务。
- 环境创建成功，Torch 的 CPU / CUDA 构建与目标一致。
- CPU toy 跑通，有训练前后独立评估。
- GPU 诊断与真实 SFT 反向传播通过，记录实际显存。
- 能从 rollout 解释奖励、优势和更新，或解释零更新原因。
- 能区分模型缓存、LoRA 检查点、优化器状态和评估文件。

达到这些标准就完成了本地流程入门，无需先追求 GSM8K 高分或完整论文复现。

<a id="reference"></a>

## 17. 命令速查与参考资料

### 17.1 第二天重新开电脑，怎么接着学

不用重新下载源码，也不用每次 `uv sync`。正常情况是：

1. 打开 PowerShell。
2. 进入已有项目。
3. 查看旧实验，或用新输出目录开始下一项实验。

**输入：**

```powershell
Set-Location "E:\Agentic_RL\agentic-rl-lab"
```

需要重建或调整依赖时才重新执行安装步骤；需要下载缺失模型时才运行下载命令；要恢复训练进度时明确使用第 12 节的检查点。单纯重新打开终端不会丢掉硬盘中的环境和结果。

### 17.2 常用入口

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

### 17.3 第一次不用背的词典

| 术语 | 在本教程里是什么意思 |
| --- | --- |
| 环境 / venv | 项目专用 Python 与依赖集合 |
| 权重 / 模型参数 | 模型内部影响输出的数值 |
| 基模 / base | 微调前已经训练好的原始模型 |
| LoRA / adapter | 加在基模上的少量可训练矩阵及其保存文件 |
| 推理 / inference | 用现有参数产生回答 |
| 微调 / fine-tuning | 在已有模型上继续学习 |
| SFT | 用参考答案进行监督微调 |
| RL | 根据奖励信号学习策略 |
| GRPO | 本教程用同题多回答的相对奖励训练策略 |
| policy / 策略 | 决定接下来选择哪个动作或 token 的模型 |
| prompt | 模型开始回答前看到的题目与指令 |
| token / tokenizer | 模型处理文字的单位 / 负责转换的工具 |
| rollout | 一次实际生成的回答或交互轨迹 |
| reward | 按规则给回答打的分 |
| advantage | 回答相对组内平均水平的优势 |
| loss | 用来求导并优化的目标值，不等同于奖励 |
| gradient / 梯度 | 参数变化如何影响损失的局部信息 |
| backward / 反向传播 | 计算梯度的过程 |
| optimizer / 优化器 | 根据梯度执行参数调整的程序 |
| checkpoint | 权重与训练状态存档 |
| eval | 在保留题目上测验，通常不更新参数 |
| observation | 工具返回给模型的信息 |
| CUDA / 显存 | NVIDIA 的计算体系 / 显卡使用的内存 |

### 17.4 参考资料

这些资料供需要深入时查阅，不必全部读完才开始本教程。镜像服务与外部页面可能变化：

- [PowerShell 切换目录](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/set-location?view=powershell-5.1)、[读取 JSON](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/convertfrom-json?view=powershell-5.1)
- [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)、[环境变量](https://docs.astral.sh/uv/reference/environment/)、[配置优先级](https://docs.astral.sh/uv/concepts/configuration-files/)
- [清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [中科大 GitHub Release 镜像](https://mirrors.ustc.edu.cn/help/github-release.html)
- [南京大学 PyTorch 镜像](https://help.mirror.nju.edu.cn/pytorch/)
- [PyTorch 历史版本与 CUDA 轮子](https://pytorch.org/get-started/previous-versions/)
- [Qwen2.5-0.5B-Instruct 模型页](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [HF-Mirror](https://hf-mirror.com/)，可选第三方下载镜像
- [上游在线教程](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/)，用于对照原远程版本
