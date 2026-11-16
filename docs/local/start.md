# Windows 本地训练：从这里开始

完整入门教程已统一到[本地训练入门](../../experiments/getting-started/readme.md)。适用 Windows、RTX 3070 8GB、16GB 内存，使用本仓库 `local_rl`，不需要付费 API。

按你当前进度选择：

| 目前状态 | 阅读位置 |
| --- | --- |
| 已下载分支 ZIP，还未解压 | [准备项目目录](../../experiments/getting-started/readme.md#project) |
| 旧 uv 环境在备份目录 | [新环境与缓存](../../experiments/getting-started/readme.md#environment) |
| 希望新环境使用国内镜像 | [镜像与依赖安装](../../experiments/getting-started/readme.md#mirrors) |
| 已经装好环境 | [CUDA 诊断](../../experiments/getting-started/readme.md#cuda) |
| 先学习算法，不下载模型 | [CPU toy](../../experiments/getting-started/readme.md#toy) |
| 开始真实模型训练 | [模型准备与 GPU 检查](../../experiments/getting-started/readme.md#model) |
| 已完成一次训练 | [日志解读](../../experiments/getting-started/readme.md#logs)、[独立评估](../../experiments/getting-started/readme.md#evaluation) |
| 中断、报错或显存不足 | [续训](../../experiments/getting-started/readme.md#checkpoints)、[排错](../../experiments/getting-started/readme.md#troubleshooting) |

也可下载 [HTML 离线阅读版](getting-started.html)，用浏览器打开。正文与命令无需联网即可阅读，文末外部参考链接需要网络。

已安装好的本地版环境无需因这次文档更新重建。所有命令从项目根目录运行；Linux 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`，安装 / 镜像配置请使用对应 shell 语法。

进一步学习：[按代码理解 GRPO](walkthrough.md)、[迁移范围](migration.md)、[验证记录](validation.md)。
