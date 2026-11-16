# 验证记录

日期：2026-11-16。以下是执行过的检查，不是 3070 的速度或显存承诺。

## 环境与自动化测试

- Linux x86_64，Python 3.12.14。
- PyTorch 2.6.0+cpu、Transformers 4.51.3、PEFT 0.15.2、Accelerate 1.6.0。
- `uv sync --python 3.12 --extra cpu --frozen`：完成独立环境安装。
- `pytest -q`：36 项通过。
- `ruff check local_rl tests`：通过；`uv lock --check`：通过。
- 自动化测试不下载模型：在临时目录创建微型 Qwen2 和 tokenizer，再使用真实 Transformers/PEFT 训练。

覆盖：优势退化组、裁剪方向、GSPO 梯度、KL、reverse-KL 系数停止梯度、数值奖励、数据分割、受限计算器、生成 token 对齐、工具观察不计入 targets、冻结参考/教师、LoRA 保存恢复、无信号时跳过更新、DAPO 采样预算、断点续训逐参数一致性。

仓库另提供 Windows/Linux CPU 的 GitHub Actions 工作流。这里的本地测试环境没有 CUDA；下文另记录读者提供的 Windows CUDA 日志片段，完整实验的峰值、速度与稳定性仍需实机验证。

## 离线 toy 实验

```bash
python -m local_rl.train --backend toy --task toy --device cpu --steps 100 --batch-size 4 --group-size 8 --learning-rate .01 --save-every 100 --output outputs/toy-max-100
```

任务：两个 0–9 数字中选出较大者；十个离散动作；从随机初始化的 1,994 参数网络开始。

| 项目 | 观察值 |
| --- | --- |
| seed | 42 |
| 训练样本 | 固定训练分区的前 64 题 |
| 评估样本 | 独立分区 20 题，greedy |
| 初始评估正确率 | 2/20 = 10% |
| 训练后评估正确率 | 11/20 = 55% |
| 采样轮数 / 优化次数 | 100 / 85 |
| 全程采样平均奖励 | 0.5015625 |

这是一次固定种子的小实验，只能说明该教学任务出现了学习信号，不代表语言模型能力、统计显著性或所有种子都会提升。

## 真实 Qwen2.5-0.5B-Instruct

模型 revision：`7ae557604adf67be50417f59c2c2f167def9a775`。CPU FP32，LoRA rank 8，`q_proj` / `v_proj`，540,672 个可训练参数。公开权重下载一次后，以下执行使用 `--offline`。

| 检查 | 实际观察 |
| --- | --- |
| SFT，2 轮，8 个候选算术训练题 | 完成 2 次更新并保存适配器与优化器；梯度非零 |
| GRPO，2 轮，每题 4 个回答，max-new-tokens=64 | 第一轮奖励均值 0.5、一次有效更新；第二轮奖励全零、正确跳过 |
| 基模 vs GRPO 检查点，独立 4 题，greedy | 两者严格 boxed 数值正确率均为 0/4，格式率均为 0/4 |
| 本地 OPD，SFT 适配器作教师，1 轮、1 条回答 | 完成一次学生更新，梯度范数约 0.758；保存本地检查点 |

GRPO 实际命令：

```bash
python -m local_rl.train --device cpu --offline --steps 2 --group-size 4 --max-new-tokens 64 --output outputs/qwen-grpo
```

评估中部分回答算对了数字，但没有按要求输出 `\boxed{...}`，因此严格奖励为零。这正是需要同时看原始文本和格式率的原因。**没有证据表明这两轮 GRPO 提升了真实模型效果。** 短程检查验证的是生成、评分、求导、更新、保存与评估能贯通。

## 未验证与未实现

- 已收到一段 Windows CUDA 读者日志（见下文），仍缺完整实验的 GPU 峰值、吞吐与长时间稳定性数据。
- 尚无 Windows CUDA 完整训练实测。
- 真实 GSM8K 训练、长期提升、多随机种子统计尚未执行。
- 本地教师不保证比学生强；工具调用不保证模型学会使用。
- QLoRA、完整 DAPO 长度奖励、大型多模态、ALFWorld/TEMPO/AgentOPSD/Tournament 的全量迁移不在当前实现内。
- 请对照[迁移表](migration.md)，不要把教学变体描述成完整论文复现。

## 初版教程命令核对（2026-11-16）

- `experiments/getting-started/readme.md` 已改为 17 节 Windows 本地课程；国内镜像、ZIP 解压与旧环境处理均单独说明。
- 核对 24 条 Python CLI 命令，其中 13 条训练命令：参数解析、检查点来源和直接 GRPO 续训配置均通过检查。此项是静态核对，不表示执行过 Windows CUDA 命令。
- 按教程参数重新执行 CPU toy 100 轮、摘要与两次独立评估（只将输出改到临时目录）：初始 2/20，训练后 11/20。
- HTML 阅读版由 Markdown 生成，17 个内部目录锚点及源码相对链接通过检查；正文不依赖外部字体、脚本或样式。当前执行环境无法启动 Chromium，未宣称完成浏览器视觉验证。
- 教程与阅读版未改变训练代码、核心依赖或锁文件。更新文档无需重建已安装好的本地版环境。

当前续训限制：从 `--adapter` 热启动的实验，原 adapter 配置校验与 CLI 的 resume/adapter 互斥条件冲突，不能直接精确恢复。可加载最新 adapter 开始新实验，但优化器和随机状态重置。直接从基模开始的 GRPO 续训示例已核对，不受此限制。

## Windows CUDA 读者反馈片段（2026-11-16）

证据来源是读者贴出的命令与局部终端日志，非本环境亲自运行。读者设备背景为 RTX 3070 8GB、16GB 内存；片段未包含完整 doctor 报告、20 轮结束状态或独立评估。

命令：`local_rl.train --task arithmetic --device cuda --steps 20 --batch-size 1 --group-size 4 --max-new-tokens 64 --offline --output outputs/grpo-20`。

第 1 轮报告：`optimizer_updates=1`、`reward=0.5`、`active_groups=1`、`loss=0.0`、`grad_norm=6.619186878204346`、`mean_tokens=22.0`、`truncated_fraction=0.25`、`seconds=9.84016000002157`、`peak_allocated_mib=1089.68896484375`、`peak_reserved_mib=1192.0`。可训练参数 540,672，训练池 64 题。

这支持“至少一轮本地 CUDA 更新已经运行”的判断，不支持完整训练耗时、全程最大显存或效果改善的结论。教程仅将这一轮作为日志阅读例子，并标明其 64-token 配置与正文 96-token 示例不同。

## 零基础详细版核对（2026-11-16）

- 扩展为第 0–17 节（共 18 节），新增终端操作、鼠标解压、逐步安装检查、命令拆解、奖励手算、逐项日志解读、读评估文件、数据文件创建和自测。
- 24 条 Python CLI 命令（含 13 条训练命令）完成参数解析、检查点引用和续训配置核对；并移除了 SFT 评估文件的重复写入步骤。
- CPU toy 按正文设置重新跑完训练、摘要与两次评估，仅替换临时输出位置，仍为 2/20 → 11/20。
- 新增 JSONL 示例经过真实 `load_examples` 读取：4 道 train、4 道 eval，题目不重合；示例 JSON 格式有效。
- HTML 的 53 个代码块与 Markdown 逐字符核对一致，18 个目录锚点有效；新增命令 / 输出标签，内嵌 JavaScript 语法检查通过。正文无外部字体、脚本或样式依赖。
- 文档生成脚本 Ruff 与 Python 编译检查通过；未修改训练代码与项目依赖。本轮没有声称执行 Windows PowerShell 或 GPU 全量实验，也没有新增浏览器视觉验证。
