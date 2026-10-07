# GujinBridge 数据集 V5 翻译任务抽样审核报告（2026-10-05）

## 结论

V5 对三类非确定性风险进行了分层抽样审核，共 261 条，现已全部完成，没有待定项：

| 任务 | 审核数 | 通过 | 拒绝 | 拒绝率 |
|---|---:|---:|---:|---:|
| 古译今 `c2m` | 150 | 95 | 55 | 36.67% |
| 今译古 `m2c` | 111 | 58 | 53 | 47.75% |
| 合计 | 261 | 153 | 108 | 41.38% |

218 条为高置信判定，43 条为中等置信判定。24 条通过样本修订了参考答案。

本轮是风险样本的分层抽样审核，用于估计审计规则的有效性；它不是随机抽取的全数据质量估计，也不能作为模型评测集。

## 按审计标记统计

同一记录可能同时命中多个标记，因此下表各行不可相加：

| 标记 | 样本数 | 通过 | 拒绝 | 拒绝率 |
|---|---:|---:|---:|---:|
| `low_character_overlap` | 107 | 21 | 86 | 80.37% |
| `number_or_date_mismatch` | 117 | 76 | 41 | 35.04% |
| `conflicting_duplicate_prompt` | 61 | 58 | 3 | 4.92% |

由此确定以下策略：

1. `conflicting_duplicate_prompt` 不能整类删除。绝大多数单条答案本身正确；应仅删除重复组中明确错配或额外混入其他事件的答案。
2. `number_or_date_mismatch` 不能整类删除。大量样本只是把“十二日”转换为“丙子”一类干支纪日；人物、地点、月份和数量不一致时才应删除或复核。
3. `low_character_overlap` 是强风险信号，但仍有 19.63% 的抽样记录有效，不能仅凭该标记自动删除。应优先审核全部剩余低重合记录，再处理日期数字类记录。

## 三条疑难古译今记录

此前 3 条 `needs_review` 已回查原典并定稿：

- `instruct#728923`：依据《宋史》奎宿条确认“犯之”“舍”均指奎宿，修正“鲁园”“动擂”等讹字后通过；
- `instruct#473133`：依据《太平广记》女灵观故事，删除参考中混入的前句情节，只保留当前句译文后通过；
- `instruct#489587`：依据《太平广记》“望江李令”全文确认“之”指两个儿子，修订译文后通过。

核对来源：

- 《宋史·志·卷四》：https://www.alinkbc.com/guidang/21110.html
- 《太平广记》女灵观上下文：https://xiaozhiliaoo.github.io/reading-note/guoxue/materials/zhcscs/xs09j.pdf
- 《太平广记·卷第三百五十三》：https://zh.wikisource.org/zh-hans/太平廣記/卷第353

## 输出文件

- `data/gujinbridge/audit/reviews/gpt56_v5_c2m.jsonl`：150 条古译今审核，其中 3 条最终定稿由 `codex-gpt-5` 回查原典完成；
- `data/gujinbridge/audit/reviews/codex_v5_m2c.jsonl`：111 条今译古审核；
- `data/gujinbridge/audit/v5_reviewed.jsonl`：合并后的 261 条完整审核记录；
- `data/gujinbridge/audit/v5_review_report.json`：机器可读审核统计。

## 下一步

前三项已经完成，并生成 `data/gujinbridge/processed_v5_reviewed/` 中间版本：

- 从 102,334 条降至 102,226 条；
- 删除 108 条明确错误记录；
- 修订 24 条参考答案；
- 严格审计记录由 4,009 条降至 3,886 条；
- `low_character_overlap` 从 1,188 条降至 1,098 条；
- `number_or_date_mismatch` 从 2,581 条降至 2,535 条；
- `conflicting_duplicate_prompt` 从 346 条降至 337 条；
- 训练、验证和测试仍保持来源隔离。

变更明细位于：

- `data/gujinbridge/processed_v5_reviewed/removed_by_review.jsonl`；
- `data/gujinbridge/processed_v5_reviewed/corrected_by_review.jsonl`；
- `data/gujinbridge/processed_v5_reviewed/manifest.json`；
- `data/gujinbridge/audit/v5_reviewed_report.json`；
- `data/gujinbridge/audit/v5_reviewed_flagged.jsonl`。

## 最终保守清洗与重训前状态

为避免在 3,886 条剩余风险记录上继续引入主观批量判定，最终版本采用保守策略：

- 保留 139 条已有明确批准结论的风险样本；
- 删除其余 3,747 条未审核风险样本；
- 保留全部 243 条 V4 高置信黄金评测记录，并确认它们只位于测试集；
- 最终剩余 98,479 条，其中训练 95,882、验证 1,157、测试 1,440；
- 最终审计仍提示 104 条，但它们全部属于已明确批准的审核例外，未解决风险为 0；
- 标点任务最终为训练 2,998、验证 145、测试 361 条。

全量 tokenizer 预检使用 `Qwen/Qwen3-1.7B` 对应聊天模板：

| Split | P50 | P95 | P99 | 最大 | 超过 1024 |
|---|---:|---:|---:|---:|---:|
| Train | 141 | 231 | 462 | 804 | 0 |
| Validation | 145 | 457 | 490 | 585 | 0 |
| Test | 151 | 444 | 467 | 496 | 0 |

最终数据位于 `data/gujinbridge/processed_v5_final/`，训练前报告位于
`data/gujinbridge/processed_v5_final/preflight_report.json`。所有结构、全局 ID 唯一性、来源隔离、
任务覆盖、黄金集位置、离线 Base Model、聊天模板和上下文长度检查均已通过。

专用训练启动器已经准备好，但本阶段没有启动训练：

```powershell
# 先运行 20 step 冒烟训练
.\scripts\run_sft_gujinbridge_v5.ps1 -SmokeTest

# 冒烟训练通过后再运行正式 V5 重训
.\scripts\run_sft_gujinbridge_v5.ps1
```
