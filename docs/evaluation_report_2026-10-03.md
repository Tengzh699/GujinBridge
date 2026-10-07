# GujinBridge SFT 评测报告（2026-10-03）

## 评测设置

- Base Model：`Qwen/Qwen3-1.7B`
- LoRA：`outputs-gujinbridge-sft-qwen3-1.7b`
- 测试集：1,078 条，按 `source` 与训练集隔离
- 任务构成：`c2m=659`、`m2c=419`、`punctuate=0`
- 解码：贪心解码（`temperature=0`），最多生成 128 tokens
- Qwen3 thinking：关闭
- 系统提示词：`configs/gujinbridge_system_prompt.txt`
- 指标：去除空白后的严格匹配、字符集合 Precision/Recall/F1

字符指标适合做同一测试集上的回归比较，不能取代人工忠实度、流畅度与出处核验。

## Base 与 LoRA 对比

| 指标 | Base | GujinBridge LoRA | 变化 |
|---|---:|---:|---:|
| Overall Exact Match | 0.09% | 1.21% | +1.11 个百分点 |
| Overall Char Precision | 0.2152 | 0.6872 | +0.4720 |
| Overall Char Recall | 0.7054 | 0.6268 | -0.0787 |
| Overall Char F1 | 0.2939 | 0.6467 | **+0.3528（+120.0%）** |
| `c2m` Char F1 | 0.2336 | 0.6256 | **+0.3920（+167.8%）** |
| `m2c` Char F1 | 0.3889 | 0.6799 | **+0.2910（+74.8%）** |

Base 的字符召回率较高不是优势：它经常复述题目并补充解释，从而碰巧覆盖更多参考答案字符，但精确率很低。
LoRA 输出更接近目标答案的长度和格式，因此 Precision 与 F1 显著提高。

## 逐样本与输出长度

| 统计项 | 结果 |
|---|---:|
| LoRA F1 胜出 | 1,034 / 1,078（95.92%） |
| 持平 | 9 / 1,078 |
| Base F1 胜出 | 35 / 1,078 |
| Base 平均输出长度 | 123.2 字符 |
| LoRA 平均输出长度 | 24.8 字符 |
| 参考答案平均长度 | 28.2 字符 |
| Base 输出超过 100 字符 | 74.86% |
| LoRA 输出超过 100 字符 | 0.19% |

微调不仅提升了答案重合度，也显著抑制了无关解释和格式扩张。LoRA 平均输出长度与参考答案更接近。

## 数据质量发现

测试集中存在少量明显的“问题与参考答案错配”，会压低所有模型的指标。例如：

- `instruct#631811`：输入为“乾宇晏，地区谧”，参考答案却是“祭器陈列，备具礼容。传。”；
- `instruct#1838462`：输入是隋代礼乐任命记录，参考答案却是另一段“家无完堵……”；
- `instruct#1838435`：输入要求解释“见善必进……”，参考答案却是大赦、流放官员的叙述。

此外，本次 `punctuate` 的 20,000 条样本全部进入训练集，验证集和测试集没有断句任务。因此本报告只能证明古今互译能力提升，不能证明断句能力。

## 结论

本次 LoRA SFT 有明确效果：总体字符 F1 超过基座模型一倍，并且 95.92% 的测试样本优于基座。
当前模型适合进入人工抽检、数据清洗和独立断句评测阶段，但在修复错配样本与任务切分前，不建议把现有数字作为正式发布成绩。

原始结果位于：

- `data/gujinbridge/eval/metrics_base.json`
- `data/gujinbridge/eval/metrics_lora.json`
- `data/gujinbridge/eval/predictions_base.jsonl`
- `data/gujinbridge/eval/predictions_lora.jsonl`
