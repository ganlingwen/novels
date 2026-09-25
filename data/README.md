# 本地 SFT / DPO 数据

`schema.json` 是可执行的 JSON Schema（Draft 2020-12），`real/*.json` 和
`synthesized/*.json` 统一采用版本 `3.0`。
保存新编辑记录时遵循 [training-data.md](../skills/training-data.md)。

## 统一结构

每个文件只有 `schema_version`、`metadata`、`records` 三个根字段。
每条 record 都有 `id`、`prompt`、`sft`、`dpo`、`metadata`：

| 字段 | 用途 |
| --- | --- |
| `id` | 全数据集唯一的记录 ID，也是 SFT ID |
| `prompt.user_request/context/original_text` | 显式任务、字符串上下文、原文；后两者可为 null |
| `sft.eligible/response` | 是否导出 SFT，以及作者最终接受的文本 |
| `dpo.eligible/chosen_candidate_id/candidates` | 是否导出 DPO、明确选择的候选、所有真实候选 |
| `candidate.candidate_id/response/status/pair_id` | 候选文本和历史状态；可训练的 rejected 候选有唯一 pair_id，其他候选为 null |
| 各层 `metadata` | 来源、review、质量、标签、选择理由、历史计数和其他原始注释 |

`synthesized/` 中的每条记录必须声明 `metadata.data_origin="synthetic"`，并填写
`metadata.synthetic_provenance`：父记录、来源分组、合成方法、事实源、生成模型与参数、
评审模型与结论、审核状态。合成记录不得使用 `human_accepted` 等真实作者标签。

所有候选都回答本记录的同一 prompt。每个 `status="rejected"` 的候选与明确选中的候选构成一对。
其他历史状态仍保留，但不自动推断训练资格。SFT 最终修订可能不同于 DPO 当时选择的候选。
只有明确设为 eligible 的部分参与训练；不会从 DPO 自动生成 SFT。

完整实例：
[SFT + DPO](real/pr103_ch3_ward-waking-dialogue.json)、
[仅 SFT](real/pr1_ch1_occluded-pov.json)、
[仅 reviewer DPO](real/rejected_pr1_tiangan-ability-false-positive.json)。

## 2026-09-24 迁移

迁移前共 76 个文件，存在三种主要布局：

| 原布局 | 文件数 | 迁移 |
| --- | ---: | --- |
| 根对象是一条记录 | 44 | 放入 `records[]`；其中一个负例使用已有显式 DPO projection |
| 根对象包含 `records[]` | 14 | 统一各记录字段 |
| 根级独立 `sft[]`、`dpo[]` | 18 | 每条现有样本成为一条记录，保留原训练资格，不推测合并关联 |

同时消除了 `output/response/chosen` 别名、`preference.rejected[]` 文本数组、
按 status 猜 chosen，以及从 scene/focus 动态补 prompt 的分支。
旧 loader 合成的 prompt 已原样落盘，并在 `prompt.metadata.provenance` 标明来源。
11 处对象形式 context 保留旧 loader 的字符串呈现，原对象保存在
`prompt.metadata.original_context`。正文中的真实换行和字面量 `\n` 均未转换。

原始文本、候选状态、来源、review、标签和其他元数据均保留。
根级历史注释移入文件 metadata，记录注释移入记录 metadata；旧 schema_version 和
counts 也作为历史注释保留，不能当成当前版本或当前统计。
迁移前后按文件比较了原始标量多重集，确认无值丢失。

旧 loader 实际输出 **174 SFT / 230 DPO**。其中三条无关记录共享
`final_context_confirmation`，全局 ID 去重错误丢弃了两条 SFT。
迁移为以下两条补上文件名前缀，并将旧 ID 保存在 `metadata.original_id`：

- `ch3_baichuan_medical_bill_20260919:final_context_confirmation`
- `ch3_cat_kill_credit_20260919:final_context_confirmation`

迁移完成时为 **176 SFT / 230 DPO**。此前全部 174 SFT 和 230 DPO 的
ID、来源、prompt、response/chosen/rejected 及相对顺序都逐项一致；
迁移时使用 SHA-256 摘要完成一次性核对。恢复的两条样本可能改变 SFT 训练/验证切分。

迁移后的新增数据不在本 README 逐项登记；来源、审核决定和数量贡献保存在各自 JSON 中。
当前文件数、编辑记录数、SFT 样本数和 DPO 配对数由 `python data/count.py` 动态统计，
不维护容易过期的手工总数。

## 使用与验证

安装 `llm/requirements.txt` 后，在仓库根目录执行：

```bash
python data/count.py
cd llm
python -m pytest
```

`load_local_sft_records()` 和 `load_local_dpo_records()` 默认只加载 `real`。通过
`sources=("synthesized",)` 或 `sources=("real", "synthesized")` 显式选择来源；
直接传入 `data/real/` 或 `data/synthesized/` 时，sources 必须与目录一致。
直接子目录仅允许选择该单一来源；选择两个来源时必须传入父目录 `data/`。
loader 使用仓库内 `data/schema.json` 验证格式，并检查跨字段候选引用、
全局记录/pair ID 唯一性、chosen/rejected 差异。错误包含文件及记录位置，不再静默跳过。
合成记录只要 SFT 或 DPO 任一 eligible=true，judge.decision 就必须为 accepted；
rejected/tie 记录只能在两个 eligible 均为 false 时保留为档案。
`count.py` 独立统计显式偏好注释并对比 loader 输出。

`sample_training_records(records, {"real": 1, "synthetic": 1}, sample_count, seed)`
按来源权重采样，再在该来源内均匀采样。权重是相对概率；`1:1` 表示两类来源各约 50%，
不是每条记录同权。验证集不要经过此采样。

有意新增或修改语料后，应审查并更新 `llm/tests/test_local_data.py` 的数量和语料摘要断言。

## 作者审核中的正文修改

`review/*.json` 保存尚未接受的真实正文编辑提案，使用同一 schema，所有 eligibility 为 false。
该目录不由训练 loader 加载；审核前不改正文，不给候选标 chosen/rejected。
作者接受修订后再应用并迁入 `real/`；DPO 还需要明确的同题候选优劣判断。
分类与一致率口径见 [editorial_preferences.md](editorial_preferences.md)。
