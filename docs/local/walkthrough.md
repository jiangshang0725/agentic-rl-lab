# 从一组回答到一次参数更新

阅读顺序：`data.py` → `policy.py` → `losses.py` → `train.py` → `checkpoint.py` → `evaluate.py`。下面按实际执行顺序解释关键语句；这些函数可直接下断点观察，不经过远程训练 SDK。

## 1. 题目、答案与策略

`load_examples(task, "train")` 返回题目和答案；固定分割由独立的随机种子 1729 决定，训练随机种子不会改变评估集。

- `ToyPolicy` 是从随机参数开始的十选一网络。两个数字分别变成 one-hot 特征，经 MLP 输出十个动作的 logits。
- `HFPolicy` 加载已有 Qwen 权重，用 PEFT 添加 `q_proj`、`v_proj` 的 LoRA。基础模型冻结，优化器只接收 `requires_grad=True` 的参数。
- `model.generate()` 只负责生成，不保留反向传播图。训练时再对已生成的 token 做前向计算，得到可求导的 log probability。

LoRA 不是“假训练”：更新的是附加低秩矩阵。模型的回答分布仍然随这些参数变化。它减少了梯度和优化器状态，但不会消除基础模型或中间激活的显存。

## 2. 同题多次采样

`group = [policy.rollout(...) for _ in range(group_size)]` 使用当前策略回答同一道题 G 次。整组采样、旧概率、教师或参考概率计算结束后才更新权重。

基础实现逐条采样，避免同时保存 G 份生成缓存。`top_p=1`、`top_k=0`，训练与采样使用相同的 temperature。否则，截断采样分布与训练时重算的概率可能不一致。

学习旧策略时，先用 `torch.no_grad()` 重算生成 token 的概率并保存 `.detach().cpu()`；这保存的是数值，不是模型副本。后续多次更新也始终使用这份旧概率。

## 3. 奖励与优势

`reward(text, answer)` 读取最后一个 `\boxed{...}`，使用 Decimal 比较数值。`1,000` 与 `1000.0` 可等价；单位、表达式、NaN、无框答案都不合规。没有调用裁判模型。

例如奖励为 `[0, 0, 1, 1]`，均值为 0.5，减均值后是 `[-0.5, -0.5, 0.5, 0.5]`。默认再除以总体标准差，得到 `[-1, -1, 1, 1]`：

\[
A_i = \frac{r_i - \bar r}{\max(\operatorname{std}_{population}(r), 10^{-6})}
\]

`--no-normalize-advantages` 则只减均值，更接近上游同步脚本。全零或全一奖励的优势全为零，不应产生策略更新。`steps` 统计采样轮次，`optimizer_updates` 统计真正执行的优化次数。

## 4. token 对齐与工具观察

假设 prompt 为 `[p0, p1, p2]`，回答为 `[a0, a1, eos]`：

- 输入模型的是 `[p0, p1, p2, a0, a1]`；
- `p2` 位置的输出预测 `a0`；
- `a0` 位置预测 `a1`，`a1` 位置预测 EOS。

因此 `_action_logps()` 从 `len(context)-1` 开始取 logits，再取回答 token 作为 target。prompt 不作为训练目标；EOS 是模型生成的动作，必须计入。

工具交互将轨迹拆成多个 `Action(context, tokens)`。工具返回内容进入下一次的 context，但不会进入任何 Action 的 tokens。因此既能让下一轮看到工具结果，也不会把工具的确定性输出错误地当成模型生成概率训练。这里没有 padding，因为每次只处理一条轨迹。

## 5. 策略损失

当前与旧策略对同一 token 的比率为：

\[
\rho_t = \exp(\log \pi_\theta(a_t|s_t)-\log \pi_{old}(a_t|s_t))
\]

`old_logps.detach()` 阻止梯度流向旧概率。`policy_terms()` 实现：

| 选择 | 关键计算 | 教学含义 |
| --- | --- | --- |
| importance_sampling | `-ratio * advantage` | 提高正优势回答的概率 |
| grpo / ppo | `-min(ratio*A, clip(ratio)*A)` | 限制过大的策略改变；没有额外 value critic |
| cispo | `-detach(clamp(ratio))*logp*A` | 截断权重并在该权重处停止梯度 |
| gspo | 先对整条回答的 log-ratio 求平均再 exp | 按序列比率裁剪，而非分别裁剪每个 token |
| dapo | 高低阈值不对称 + 全部动作 token 的均值 | 学习 token 归一化与动态采样 |

GRPO/GSPO 先对一条回答取均值，再对整批回答取均值。DAPO 对被选中组的所有动作 token 取全局均值，长短回答权重因此不同。每条轨迹立即 `backward()`，无需把整批计算图同时保存在显存。

默认 `updates_per_batch=1`：更新前当前与旧策略相同，比率为 1，clipping 通常不触发。要观察裁剪可增加更新次数，但要监控梯度与评估；OPD 只允许一次更新。

初始 loss 可能等于零，因为优势均值为零，但其对每条回答的导数并不等于零。要同时看 `grad_norm`、`optimizer_updates` 和评估结果。

## 6. 可选 KL 和 OPD

`--beta 0.02` 会加入对冻结基模的 KL 估计。通过临时禁用 LoRA 得到基模概率，不额外复制参考模型。这个参考始终是原始基模；如果从 SFT 适配器开始，它不是 SFT 检查点。默认 beta 为 0，并不声称默认配置完全等同原始带 KL 的 GRPO。

OPD 读取本地教师适配器：同一基模切换 teacher/default，教师只在 no-grad 中打分。对学生采样 token 使用：

\[
L_t = \operatorname{stopgrad}(\log \pi_{old,t}-\log \pi_{teacher,t})\log \pi_{\theta,t}
\]

这是 token 级采样 reverse-KL 的 score-function 梯度估计，系数不能反向传播；不是完整词表 KL，也不是完整轨迹 KL 的精确计算。师生必须使用同一基模和词表。教师与学生完全相同时信号为零；训练教师的质量需要另行评估。组内奖励只用于观测，不决定 OPD 的更新。SFT 的 reward 字段占位为 0，不是生成评估结果；SFT/OPD 的 active_groups 表示实际处理的组。

## 7. 更新、保存和评估

每个更新依次执行：

1. `optimizer.zero_grad(set_to_none=True)` 清除上次梯度。
2. 逐条计算 loss 并 `scaler.scale(loss).backward()` 累积梯度。
3. `scaler.unscale_(optimizer)` 还原梯度，再裁剪梯度范数至 1。
4. `scaler.step(optimizer)` 更新 LoRA 或 toy 网络，`scaler.update()` 更新混合精度缩放状态。

CUDA 上优先使用设备支持的 BF16，否则 FP16 + GradScaler；CPU 使用 FP32。检测到非有限梯度会停止，避免把坏权重继续保存。基础模型不使用 `device_map="auto"` 把训练无声地卸载到 CPU。

检查点包含适配器（toy 为完整小网络）、tokenizer、优化器、scaler、Python/Torch/CUDA RNG、配置、依赖版本、数据指纹和可获得的模型 revision。`manifest.json` 最后写入，作为保存完成标志。评估不加载优化器，用独立题集和固定 greedy 解码比较基模与检查点。

## 8. 建议实际观察的内容

- 打开 `rollouts.jsonl`，手算一个组的优势，与日志核对。
- 把组大小设成 2 与 4，观察有效组比例与每轮耗时。
- 比较全对/全错组与有奖励差异的组。
- 用相同 seed、数据和预算比较 GRPO 与 GSPO；不要混用不同训练条件。
- 将 `max-new-tokens` 降低，观察截断率如何影响奖励，而不是只看显存下降。
- 在工具任务中核对 `observations`，确认工具返回的数值不属于策略 targets。
