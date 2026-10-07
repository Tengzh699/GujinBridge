# GujinBridge 数据目录

- `demo/`：仓库内置的少量 ShareGPT 格式样本，仅用于格式检查和训练冒烟测试。
- `raw/`：下载的上游原始 JSONL；已在 `.gitignore` 中忽略。
- `processed/`：经 `tools/prepare_gujinbridge_dataset.py` 清洗、抽样并按出处切分后的数据；已忽略。
- `processed_v2/`：按任务保障验证/测试来源覆盖的新版本数据；已忽略。
- `eval/`：导出的评测问题、参考答案和模型预测；已忽略。
- `audit/`：可疑错配、重复、长度及日期异常的审计报告；已忽略。
- `gold/`：待人工逐条确认的黄金测试集候选；已忽略，审核通过后应另存为版本化 benchmark。

推荐上游数据为 [gujilab/chinese-classical-corpus](https://github.com/gujilab/chinese-classical-corpus)。
完整数据不随本仓库分发。使用或发布模型前，请复核所用数据版本、许可和内容质量。

演示集中的 `train`、`validation`、`test` 使用不同的 `source`，用于展示“同一出处不跨集合”的切分原则。
