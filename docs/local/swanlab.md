# 本地训练 + SwanLab 本地看板：Windows 操作教程

适用本地版 `feat/local-training-8gb`，Windows PowerShell、RTX 3070 8GB、16GB 内存。训练仍由 `local_rl.train` 在你的电脑上执行；SwanLab 只负责显示日志，不承担训练。无需 PyTRIO、API Key、SwanLab 账号或云服务。

## 1. 与原教程逐项对应

| 原教程命令 / 目的 | 本地版对应做法 |
| --- | --- |
| `uv sync`：安装依赖 | `uv sync --python 3.12 --extra cuda --frozen`，明确安装 CUDA 版本 |
| `uv run trio login`：登录远程训练服务 | 省略；本地 GPU 不需要登录 |
| `uv run swanlab login`：登录云端日志服务 | 省略；使用 SwanLab `local` 模式 |
| `uv run swanlab verify`：验证云端凭据 | 省略；用 `local_rl.doctor` 检查本地训练环境 |
| `experiments/grpo/01-demo-sync.py` | 改为 `python -m local_rl.train`；旧脚本仍依赖远程服务 |
| `--max-tokens` | 本地参数叫 `--max-new-tokens` |
| `--swanlab-mode online` | 不传给本地训练器；导入工具固定使用 `mode="local"` |
| `swanlab watch experiments/grpo/swanlog` | 先把本地 JSONL 转成 SwanLab 日志，再看 `outputs/swanlog` |

本地版默认生成 `metrics.jsonl`，SwanLab 不能直接把它当作自己的日志。新增的 `local_rl/swanlab_export.py` 负责转换，并可启动看板。它支持导入历史日志和跟随正在写入的日志。

**得到的是本地采样 → 奖励 → 更新参数 → 记录指标 → 看曲线的学习流程。** 模型大小、任务、实现和参数与原远程实验不同，训练速度、数值和效果不会自动相同。

## 2. 已下载 ZIP 的用户：只补充一个文件

如果你已经能运行本地训练，不需要重新解压整个项目，也不用重建训练环境。

1. 下载本分支的 [`swanlab_export.py`](../../local_rl/swanlab_export.py)。在 GitHub 文件页可使用 Download raw file；若打开了源码文字，保存文件，文件名必须是 `swanlab_export.py`。
2. 把文件放进 `E:\Agentic_RL\agentic-rl-lab\local_rl` 文件夹。
3. 文件资源管理器中打开“查看 → 显示 → 文件扩展名”，确认没有误存成 `swanlab_export.py.txt`。
4. 打开 PowerShell，依次执行下面的命令。

```powershell
cd E:\Agentic_RL\agentic-rl-lab
Test-Path .\local_rl\swanlab_export.py
```

预期显示 `True`。如果是 `False`，先修正保存位置，后面的命令暂时不要执行。

这个脚本声明了自己的可选依赖 `swanlab[dashboard]==0.10.4`，uv 首次运行时会自动准备独立的看板环境并缓存。它不会给项目 `.venv` 安装或卸载 PyTorch；无需激活环境。**运行方式是 `uv run local_rl/swanlab_export.py ...`，不要在中间加 `python`，也不要把它改成 `python -m local_rl.swanlab_export`**，否则不能利用脚本依赖自动安装功能。

## 3. 让看板依赖使用国内镜像

如果你之前已通过 `UV_DEFAULT_INDEX` 或用户级 uv 配置设置默认镜像，继续沿用。只改项目 `pyproject.toml` 的源不等于给独立脚本配置了镜像。

在当前 PowerShell 窗口显式指定清华源：

```powershell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
```

这条只作用于当前窗口。希望以后新打开的终端也使用这个默认源，可再执行一次：

```powershell
[Environment]::SetEnvironmentVariable("UV_DEFAULT_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple", "User")
```

已有项目明确指定的 PyTorch 专用索引、锁文件中的下载地址和模型权重下载来源不由这个变量全部改写；这些仍按[完整教程的镜像章节](../../experiments/getting-started/readme.md#mirrors)设置。首次安装需要联网；依赖已缓存且不清理缓存时，看板可在本机使用。

## 4. 最快看到曲线：打开已有的 grpo-20

如果已有训练目录 `outputs/grpo-20`，先检查：

```powershell
Test-Path .\outputs\grpo-20\config.json
Test-Path .\outputs\grpo-20\metrics.jsonl
```

两行都应显示 `True`，且训练至少完成过一轮。然后执行：

```powershell
uv run local_rl/swanlab_export.py outputs/grpo-20 --watch
```

这一步依次做三件事：首次安装看板依赖；读取已有的数字指标；启动本地网页服务。**不会重新训练模型**。第一次下载可能较慢，后续可复用缓存。

看到本地服务启动信息后，在 Edge / Chrome 地址栏输入：

```text
http://127.0.0.1:5092
```

`127.0.0.1` 指你自己的电脑，不是收费网站。进入项目 `agentic-rl-local`，选择实验 `grpo-20`。先找 `reward`、`optimizer_updates`、`grad_norm`、`loss` 四个指标。

**保留 PowerShell 窗口。** 它在提供网页服务，没有返回命令提示符是正常现象。停止查看时在这个窗口按 `Ctrl+C`；日志仍保留在 `outputs/swanlog`。

以后重新打开同一个看板，使用：

```powershell
uv run local_rl/swanlab_export.py --watch-only
```

不要为了重新看图反复导入。每次带训练目录导入都会新建一个看板实验，即使名称相同，仍是不同记录。`--watch-only` 不重复导入；以后还可给导入加 `--name grpo-20-review` 自定义名称。

如果你的实验实际叫 `outputs/qwen-grpo-01`，把导入命令里的 `outputs/grpo-20` 换成它。看板目录与训练目录是两类文件：

| 位置 | 内容 |
| --- | --- |
| `outputs/grpo-20` | 原始 `config.json`、`metrics.jsonl`、回答文本和模型检查点 |
| `outputs/swanlog` | 导入后生成的 SwanLab 数据库和画图文件 |

导入器不改写训练目录。备份学习成果时，模型检查点和原始日志仍应保留，不能只保存看板文件。

## 5. 从安装到第一次新训练的完整命令

如果只是看已有日志，第 4 节就够了。要开始一个新实验，按顺序执行本节。每条成功后再执行下一条；不要把 `PS E:\...>` 提示符复制进去。以下每个命令都写在一行，适合 PowerShell，无需原教程 Bash 的 `\` 换行符。

**第一步：进入项目目录，安装 / 恢复 GPU 训练依赖。**

```powershell
cd E:\Agentic_RL\agentic-rl-lab
uv sync --python 3.12 --extra cuda --frozen
```

这会按锁文件同步项目环境，不是每次从零下载。若你还需要之前的 Matplotlib 静态绘图依赖，安装命令增加 `--extra plots`。已有训练进程运行时不要同步它正在使用的环境。

**第二步：检查 CUDA。**

```powershell
uv run --extra cuda --frozen python -m local_rl.doctor
```

看到 CUDA 可用、显卡型号正确且反向传播检查成功后继续。找不到显卡时先按完整教程排错，`trio login` 和 `swanlab verify` 都不能修复本地 CUDA。

**第三步：使用已下载的默认 Qwen 模型，跑 10 轮。**

```powershell
uv run --extra cuda --frozen python -m local_rl.train --task arithmetic --device cuda --offline --steps 10 --batch-size 1 --group-size 4 --max-new-tokens 64 --output outputs/grpo-swan-01
```

这里假定你已经完成过 Qwen 本地训练，模型在缓存中。若提示离线模式下找不到模型，按[模型准备章节](../../experiments/getting-started/readme.md#model)先下载，或在已经配置好下载来源时删除本命令的 `--offline`。它不会向收费 API 发送训练请求。

输出目录必须是新的。`grpo-swan-01` 已存在时换成 `grpo-swan-02`，不要删除旧实验来腾名字。成功结束应出现 `Saved checkpoint`；10 轮时最终检查点通常是 `outputs/grpo-swan-01/checkpoint-000010`。

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--steps 10` | 10 轮采样，不保证 10 次有效参数更新 |
| `--batch-size 1` | 每轮 1 道题 |
| `--group-size 4` | 同一道题生成 4 个候选回答，比较它们的奖励 |
| `--max-new-tokens 64` | 每个回答最多新生成 64 tokens；较短，可能截断 |
| `--offline` | 只使用本机已经缓存的模型文件 |

原命令的 `4 × 8` 是每轮 32 个回答，这里是 `1 × 4 = 4` 个。512 个新 token 也比 64 长得多。先用小配置跑通并检查显存、耗时和截断比例，再逐项增大。当前默认总序列上限也是 512；不能把原 `--max-tokens 512` 机械替换后就假定能完整生成 512 个新 token。

**第四步：导入并打开曲线。**

```powershell
uv run local_rl/swanlab_export.py outputs/grpo-swan-01 --watch
```

然后打开 `http://127.0.0.1:5092`，选择 `grpo-swan-01`。

本地训练命令不支持 `--swanlab-mode`，不要附加它。看板模式由导入器明确设置为 `local`，既不需要 SwanLab 登录，也不会因为过去登录过云账号而改为在线记录。

## 6. 可选：一边训练一边看曲线

最简单的是训练结束再导入。想实时查看时，准备 **3 个 PowerShell 窗口**；每个窗口先执行 `cd E:\Agentic_RL\agentic-rl-lab`。

**窗口 A：训练。** 使用一个新的输出目录：

```powershell
uv run --extra cuda --frozen python -m local_rl.train --task arithmetic --device cuda --offline --steps 50 --batch-size 1 --group-size 4 --max-new-tokens 64 --output outputs/grpo-live-01
```

等它开始打印训练轮次，再操作窗口 B；模型还在加载时可能没有 `config.json`。

**窗口 B：持续读取新指标。**

```powershell
uv run local_rl/swanlab_export.py outputs/grpo-live-01 --follow
```

看到 `Imported ... rounds` 后再打开窗口 C。`--follow` 会等待完整的一行指标写完，再导入，不会把同一行重复计数。达到配置中的目标轮数后自动结束；若训练提前中断，它会继续等新日志，可按 `Ctrl+C` 停止。

**窗口 C：提供看板网页。**

```powershell
uv run local_rl/swanlab_export.py --watch-only
```

浏览器访问 `http://127.0.0.1:5092`，选择 `grpo-live-01`。看板按新写入的数据更新，必要时刷新页面。

在 B 或 C 按 `Ctrl+C` 不会停止 A 中的训练；在 A 按 `Ctrl+C` 才会中断训练本身。看板上的停止按钮也不负责停止原训练进程。每个训练输出目录只启动一个跟随导入器；不要同时导入和跟随同一个目录，否则会产生重复的看板实验。

`--follow` 不能和 `--watch` 同时用，实时模式按两个窗口分工。若已在窗口 C 启动服务，后续导入只用 `--follow`，不用再次启动一个占用同端口的服务。

## 7. 怎么读曲线

| 曲线 | 看什么 |
| --- | --- |
| `reward` | 此轮所有回答的平均得分；算术任务要求答案符合 `\boxed{...}` 格式且数值正确 |
| `optimizer_updates` | 累计有效更新次数；平坦表示那几轮未更新参数 |
| `active_groups` | 至少有奖励差异的组数；组内全对 / 全错时可能为 0 |
| `grad_norm` | 更新前梯度范数，帮助判断是否出现梯度信号 |
| `loss` | 用于求导的目标数值；这里接近 0 不等于没学习，也不等于能力更好 |
| `truncated_fraction` | 触及生成长度限制的回答比例；高时检查原始回答，再考虑增加长度 |
| `peak_allocated_mib` | 原训练记录的 PyTorch 活跃分配峰值，单位 MiB；不等于整卡总占用 |
| `seconds` | 每轮记录的耗时，单位秒 |

横轴使用原来的 `step`。例如续训日志从第 21 轮开始，导入后仍从 21 开始，不伪造前 20 轮。只有一两轮时可能只是几个点，长曲线需要更多真实记录。

看板显示的成功 / 中止状态是**导入任务**的状态，不证明原训练完整结束。导入器只读指标文件，不检查 checkpoint 是否写完；需另外看训练窗口和检查点文件。历史导入也无法补录未保存的 GPU 利用率、完整终端日志或训练时画面。导入器已关闭自身硬件监控，避免把“现在导入时的占用”混成“过去训练时的占用”。

训练奖励曲线不能代替独立评估；训练效果请继续做[教程第 11 节](../../experiments/getting-started/readme.md#evaluation)的训练前后对比。

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 `swanlab_export.py` | 文件不在当前项目的 `local_rl` 内，或误存为 `.py.txt`；先执行第 2 节检查 |
| `Missing ... config.json` | 输出目录写错，或训练还没有完成模型加载；等至少一轮日志出现再导入 |
| 找不到 `metrics.jsonl` / 没有可导入记录 | 训练还没完成第一轮；先看训练窗口报错 |
| 找不到 SwanLab / SwanBoard | 必须按 `uv run local_rl/swanlab_export.py ...` 直接运行脚本，让 uv 处理声明的依赖 |
| `swanlab watch` 找不到命令 | 训练环境没有装看板是正常的；用工具的 `--watch-only`，不要裸跑 `uv run swanlab watch` |
| 国内镜像暂时找不到某个版本 | 核对网络和镜像同步情况，使用已同步的 PyPI 镜像重试；保留脚本里已验证的版本 |
| 网页打不开 | 看服务终端是否报错、窗口是否仍运行；本机浏览器使用 `http` 而非 `https` |
| 5092 端口被占用 | SwanLab 可能自动选择后续端口，以它最后打印的地址为准。也可关闭之前的看板，或用 `uv run local_rl/swanlab_export.py --watch-only --port 5093` 指定其他端口 |
| 看见多个同名实验 | 导入了多次；以后重开使用 `--watch-only` |
| 曲线没有继续增长 | 普通导入只读一次；实时查看使用 `--follow`，并核对原训练仍在输出新轮次 |
| 训练结束了，跟随程序还在等 | 可能提前中断或尚未写满配置轮数；在跟随窗口按 `Ctrl+C`，检查训练日志 |
| 提示指标被截断 / 顺序错误 | 原始日志被删改或混入另一实验；保留文件调查原因，不要在训练写入时编辑 JSONL |

如果需要直接调用官方 `watch`，等价的独立环境命令为：

```powershell
uv run --no-project --python 3.12 --with "swanlab[dashboard]==0.10.4" swanlab watch outputs/swanlog --host 127.0.0.1 --port 5092
```

它只打开已经转换好的日志，不代替导入步骤。先停止其他占用 5092 的看板进程。

## 9. 使用范围与依据

- [SwanLab 官方离线看板文档](https://docs.swanlab.cn/guide_cloud/offline-board.html)：需要 `dashboard` 扩展和 `local` 模式；官方将这一看板列为历史功能，目前只做简单维护。此方案适合学习观察曲线，不要求部署云平台或 Docker。
- [uv 独立脚本文档](https://docs.astral.sh/uv/guides/scripts/)：脚本内声明的依赖由 uv 单独管理；本次没有改动项目依赖和训练器。
- [本项目验证记录](validation.md)：查看实际测试范围。CPU 功能测试不等于 Windows + 3070 完整训练实测。
