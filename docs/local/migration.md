# 原项目到本地教学版的迁移范围

默认依赖和可运行路径已改为 `local_rl/`。`experiments/` 保留上游源代码、图表、结论和引用，不能把那些远程结果当成本地测试结果。它们的安装环境隔离到 `remote/`；旧命令需加 `--project remote`。

| 原主题 | 本地对应 | 实现边界 |
| --- | --- | --- |
| 环境与基础概念 | `doctor`、Windows 启动文档、toy 网络 | CUDA 12.4 / CPU 分开选；无 API key |
| Loss Functions | `local_rl/losses.py` | IS、PPO surrogate、CISPO；不是完整 actor-critic PPO |
| GRPO | `train --objective grpo` | 同题分组、奖励、可选标准差归一化、clipped loss、可选基模 KL |
| GSPO | `train --objective gspo` | 序列比率与梯度；不声称复现论文训练结果 |
| DAPO | `train --objective dapo` | 非对称裁剪、token mean、有上限的动态采样；截断轨迹奖励置零，未实现论文完整 soft overlong shaping |
| Medical SFT / 分类前置 SFT | `train --objective sft --task jsonl` | 通用数值答案格式学习；不等同医疗或 Jev 分类评测 |
| General OPD / OPSD | `train --objective opd --teacher-adapter ...` | 冻结本地教师适配器、同基模、token 采样 reverse-KL；没有大教师或自动自进化流程 |
| Search-R1 | `train --task retrieval` | 离线合成记录、两轮协议、观察屏蔽；不是网页搜索或原数据集复现 |
| ReTool | `train --task calculator` | 受限算术 AST，不能执行任意 Python；适合学习工具交互 |
| ALFWorld | 原实现保留 | 未接入本地真实 TextWorld 环境；不能用计算器演示代替宣称完成 |
| AgentOPSD / TEMPO | 原实现保留 | 未实现其完整 turn-level teacher、macro-step critic 和环境协议 |
| Vision GRPO / Spec-o3 | 原实现保留 | 未提供视觉编码、图像数据及光谱工具的本地迁移 |
| PyTrio-Jev | 原实现保留 | 未迁移分类校准与专用评测 |
| Tournament-GRPO | 原实现保留 | 未迁移报告锦标赛、大模型裁判与在线检索 |

## 为什么不把 PyTRIO 的 import 简单替换掉

原 API 将采样、自动求导、优化器、权重快照和异步任务放在服务端。只改导入或 URL，无法得到本地计算图。这里建立显式本地训练器，供核心算法与短任务共用；原章节作为算法和实验资料对照。

## 免费与离线的边界

- `local_rl` 不调用 PyTRIO、OpenAI、DeepSeek、Jina、在线裁判或在线日志。
- 模型和 GSM8K 可从公开仓库下载，不使用其托管推理服务。
- `--offline` 要求已有模型缓存/本地目录，并使用内置任务或 JSONL；GSM8K 请先导出为 JSONL。
- 网络/磁盘/电费仍由本机承担；“无需付费 API”不等于不消耗计算资源。
- 未迁移的历史脚本可能继续请求远程服务，入口已标注，请勿将其用于无付费 API 的路径。

## 8GB 配置选择

推荐 0.5B + LoRA，context <=512、completion <=96、每题 4 个回答、单条 backward。大词表 logits、长上下文激活和生成缓存都占显存，参数量不是唯一因素。增加组数主要增加总生成工作；逐条执行不会同时驻留 G 份模型。

1.5B/QLoRA 可以作为后续实验，但当前没有提供或验证量化后端。不将 3B/4B、长推理或全量微调标成“3070 稳定可跑”。CPU 测试只能验证逻辑，不能证明该显卡上的性能。
