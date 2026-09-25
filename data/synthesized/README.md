# 合成训练数据

本目录只存放合成的 SFT/DPO JSON。格式与 `../real/` 相同，并遵循 `../schema.json`。

每条记录必须：

- 设置 `metadata.data_origin` 为 `synthetic`；
- 完整填写 `metadata.synthetic_provenance`；
- 使用全数据集唯一的 record ID 和 DPO pair ID；
- 保留所依据的真实父记录及事实源；
- 不使用 `human_accepted`、`explicit_author_choice` 等真实人工决策标签。

文件按任务类型和批次命名，例如 `dialogue_information_order_batch001.json`。
