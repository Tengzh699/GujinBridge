# GujinBridge GRPO V2 实验方案

## 实验动机

GRPO V1 能稳定完成训练，但在 361 条标点测试集上没有超过同提示词、同解码参数的 DPO 对照组。

| 模型 | 标点类型+位置 F1 | 仅位置 F1 | 原文保留率 |
|---|---:|---:|---:|
| DPO V1 controlled | 65.13% | 79.35% | 68.70% |
| GRPO V1 checkpoint-100 | 64.14% | 77.69% | 67.87% |
| 差值 | -0.98 pp | -1.66 pp | -0.83 pp |

V1 的 `exact_reward` 始终为 0，`direct_format_reward` 始终为 1；两者几乎不产生排序信号。字符 F1
还会忽略标点所在的位置，因此不适合作为该任务的主要指标。

## V2 假设

V2 使用三个可验证奖励：

1. `preservation_guard_reward`（权重 0.45）：原文字词完全保留得 1；任意增删改得 -1。
2. `typed_boundary_reward_v2`（权重 0.40）：标点类型和插入位置都正确才计分。
3. `position_boundary_reward_v2`（权重 0.15）：位置正确但标点类型错误时仍给部分信用。

标点奖励受原文保留硬门控：只要模型改动原文，两个标点奖励都归零。这样避免模型通过改写文本来获得
表面上更高的标点分数。

## 训练配置

- 初始化策略：V5 merged + DPO V1 adapter（先合并），再训练新的 GRPO LoRA。
- 数据：GRPO V1 的 800 条训练集；保持测试集来源隔离。
- 训练步数：100；每 25 步保存一次。
- 每个 prompt 采样 4 个 completion，`temperature=0.9`。
- 学习率：`1e-6`；`beta=0.01`；loss 使用 DAPO。

Smoke test：

```powershell
.\scripts\run_grpo_gujinbridge_v2.ps1 -SmokeTest
```

正式训练：

```powershell
.\scripts\run_grpo_gujinbridge_v2.ps1
```

中断续训：

```powershell
.\scripts\run_grpo_gujinbridge_v2.ps1 -Resume
```

## 评测与模型选择

训练完成后，对 checkpoint 25、50、75、100 做统一评测：

```powershell
.\scripts\eval_gujinbridge_grpo.ps1 `
  -GrpoOutput outputs-gujinbridge-grpo-v2-qwen3-1.7b `
  -OutputDir data/gujinbridge/eval_grpo_v2 `
  -Checkpoints 25,50,75,100
```

主要指标是标点类型+位置 Boundary-F1，次要指标是 Position-F1、原文保留率和 Exact Match。字符 F1
只保留为兼容性回归指标。

模型晋级标准：

- Boundary-F1 必须高于 DPO controlled；
- 原文保留率不得低于 DPO controlled；
- Exact Match 不得出现明显退化；
- 若多个 checkpoint 达标，优先选择 Boundary-F1 高且更早的 checkpoint，降低过拟合风险。

若所有 checkpoint 都不达标，则保留 DPO V1 作为发布模型，把 GRPO V2 记录为负结果，不以训练 loss
下降代替下游效果提升。

## Smoke test 结果

3 步 smoke test 已通过：模型加载、DPO 合并、GRPO 反向传播、checkpoint 与最终 adapter 保存均正常。
三个 step 的 `frac_reward_zero_std` 均为 0；标点类型奖励和位置奖励均出现组内方差，说明 V2 奖励能够
区分同一 prompt 下的候选输出。

## 正式实验结果

正式训练已完成 100/100 步，完整评测见
[`data/gujinbridge/eval_grpo_v2/SUMMARY.md`](../data/gujinbridge/eval_grpo_v2/SUMMARY.md)。

最佳实验 checkpoint 为 checkpoint-50，但其 Boundary-F1 为 64.81%，仍低于 DPO controlled 的
65.13%；原文保留率为 68.42%，也略低于 DPO 的 68.70%。因此 GRPO V2 不晋级为发布模型，发布模型
继续使用 DPO V1。
