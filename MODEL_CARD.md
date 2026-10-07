# GujinBridge DPO V1 Model Card

## 模型概览

- 项目：GujinBridge（古今桥）
- 基座：Qwen3-1.7B
- 发布栈：Qwen3-1.7B -> V5 SFT merged -> DPO V1 LoRA
- 参数高效训练：LoRA，rank 16，alpha 32，dropout 0.05
- 主要语言：中文、文言文
- 当前状态：[Hugging Face 已发布的合并推理模型](https://huggingface.co/tzh699/GujinBridge-Qwen3-1.7B-DPO)

## 支持任务

- 古文翻译为现代汉语（c2m）
- 现代汉语改写为文言文（m2c）
- 古文断句与添加现代标点（punctuate）

## 选型依据

DPO V1 在 1,440 条来源隔离测试集上取得 73.39% 总体字符 F1，相对 V5 SFT 提高 0.71 个百分点；
标点原文保留率从 64.54% 提高到 71.19%。V6 定向 SFT 和 GRPO V1/V2 均没有形成稳定综合优势，故未
替换 DPO。

| 任务 | 样本数 | Exact Match | 字符 F1 |
|---|---:|---:|---:|
| 全部 | 1,440 | 2.92% | 73.39% |
| c2m | 685 | 0.44% | 63.85% |
| m2c | 394 | 1.27% | 68.67% |
| punctuate | 361 | 9.42% | 96.65% |

标点任务原文完全保留 257/361（71.19%）。字符 F1 会因输入输出共享大量汉字而偏高，不能代替
Boundary-F1、正文保留检查或人工评价。

## 本地运行

在项目根目录执行：

```powershell
.\scripts\try_gujinbridge.ps1
```

只运行三个内置示例：

```powershell
.\scripts\try_gujinbridge.ps1 -NoInteractive
```

默认读取：

- 基础权重：`outputs-gujinbridge-v5-merged`
- 适配器：`outputs-gujinbridge-dpo-v1-qwen3-1.7b`
- 系统提示词：`configs/gujinbridge_system_prompt.txt`

## 建议用途

- 古籍数字化流程中的翻译和断句候选生成；
- 教学、检索和人工整理的辅助工具；
- SFT、DPO、GRPO 与任务专用评测的研究演示。

## 不建议用途

- 将生成内容直接作为可靠史料、校勘结论或专业注释；
- 在没有原文核验的情况下自动引用篇名、人物关系和历史事实；
- 需要唯一权威标点答案的场景；
- 医疗、法律、金融等高风险决策。

## 已知限制

- 长文本更容易出现漏字、重复或擅自改字；
- 翻译可能遗漏实体关系、时间、数字或隐含语义；
- 现代汉语改写为文言文存在多个合理答案，自动 Exact Match 较低；
- 标点具有合理多解，单参考自动指标存在偏差；
- 训练数据中的历史偏见、讹误和版本差异可能被模型继承。

## 训练数据与许可

数据主要来自 `gujilab/chinese-classical-corpus`，使用前应独立复核其当前版本和许可证。训练框架派生自
MedicalGPT，代码许可与上游署名见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [CITATION.cff](CITATION.cff)。

## 评测与复现

- 总实验报告：[docs/FINAL_EXPERIMENT_REPORT.md](docs/FINAL_EXPERIMENT_REPORT.md)
- 训练与模型选择流程：[docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md)
- GRPO V2 实验方案：[docs/gujinbridge_grpo_v2_experiment.md](docs/gujinbridge_grpo_v2_experiment.md)

