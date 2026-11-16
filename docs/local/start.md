# 零基础本地训练：按你的进度开始

完整入门教程已统一到[零基础本地训练手册](../../experiments/getting-started/readme.md)。适用 Windows、RTX 3070 8GB、16GB 内存，使用本仓库 `local_rl`，不需要付费 API。

完全没用过命令行时，先从第 0 节打开 PowerShell 开始。正文采用“本步做什么 → 在哪里输入 → 完整命令 → 预期结果 → 失败时停在哪里”的顺序，并解释新术语。

建议分四次学习：准备环境（0–5）、CPU 小实验（6）、真实模型与评估（7–11）、按需进阶（12–17）。不要求一口气读完。

按你当前进度选择：

| 目前状态 | 阅读位置 |
| --- | --- |
| 没用过 PowerShell，不懂路径 | [第 0 课：从打开终端开始](../../experiments/getting-started/readme.md#basics) |
| 已下载分支 ZIP，还未解压 | [准备项目目录](../../experiments/getting-started/readme.md#project) |
| 旧 uv 环境在备份目录 | [新环境与缓存](../../experiments/getting-started/readme.md#environment) |
| 希望新环境使用国内镜像 | [镜像与依赖安装](../../experiments/getting-started/readme.md#mirrors) |
| 已经装好环境 | [CUDA 诊断](../../experiments/getting-started/readme.md#cuda) |
| 先学习算法，不下载模型 | [CPU toy](../../experiments/getting-started/readme.md#toy) |
| 开始真实模型训练 | [模型准备与 GPU 检查](../../experiments/getting-started/readme.md#model) |
| 已完成一次训练 | [日志解读](../../experiments/getting-started/readme.md#logs)、[独立评估](../../experiments/getting-started/readme.md#evaluation) |
| 中断、报错或显存不足 | [续训](../../experiments/getting-started/readme.md#checkpoints)、[排错](../../experiments/getting-started/readme.md#troubleshooting) |

也可下载 [HTML 离线阅读版](getting-started.html)。保存后双击用 Edge / Chrome 打开；它是说明书，命令仍需复制到 PowerShell。网页会区分“复制命令”与“示例输出”。可用浏览器 `Ctrl+F` 搜索报错或术语，正文、样式与复制功能不需要外部网络；源码与参考链接需要联网。

已安装好的本地版环境无需因这次文档更新重建。所有命令从项目根目录运行；Linux 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`，安装 / 镜像配置请使用对应 shell 语法。

进一步学习：[按代码理解 GRPO](walkthrough.md)、[迁移范围](migration.md)、[验证记录](validation.md)。
