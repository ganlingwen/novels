# 编辑训练数据沉淀规范

本 skill 用于把真实小说编辑过程沉淀为可训练的 SFT 与 preference/DPO 数据。目标不是为了凑数据而制造差异，而是忠实保存“模型发现问题 → 提案 → 作者选择/拒绝 → 合并”的真实决策。

## 触发时机

当正文 review、局部修改或 PR/MR 被作者接受、拒绝，或作者要求积累训练数据时使用。正文审查本身仍遵守 `AGENTS.md` 的任务路由，先读取 `skills/write.md` 及相关事实源；本 skill 只规定如何产生、选择和保存训练记录。

## 核心原则

1. 真实编辑数据写入 `data/real/*.json`；合成数据只能写入 `data/synthesized/*.json`。每条 `records[]` 表达一个独立编辑决策，同一轮编辑可保存在同一个文件。
2. 只把作者真实接受/merge 的修改作为 SFT target；不得把模型自评“正确”当作 human accepted。
3. Preference/DPO 必须来自同一个 prompt、同一份上下文下的真实候选比较。不得为历史 accepted edit 事后编造 rejected candidate。
4. 候选版本必须在作者选择前原样保存；chosen/rejected 都保留完整文本。
5. “不修改/保留原文”是合法候选，用于训练编辑克制和减少 false positive。
6. 详细 `edit_instruction` 若由 AI review 产生，必须标记 `metadata.review.provenance=ai_discovered_during_review`，不得伪装成作者 prompt。
7. 作者只说“继续”“复审”“改下一处”等宽泛要求时，`prompt.user_request` 应忠实表达这种自主审查任务；详细问题放在 `metadata.review`，不要塞进 user prompt。
8. 保留 repo、文件、PR、commit 等 provenance，使训练记录可追溯。

## 推荐编辑流程

每次先按事实源独立判断是否真的存在问题。若不存在值得修改的问题，应明确选择 no-change，不为产生数据而硬改。

发现一个真实问题后，优先给出两个都合理、但编辑取向有真实差异的候选 A/B。例如 A 做最小修改，B 可适度重组表达。不要故意写一个明显差的 B。

作者选择 A 或 B 后：
- chosen 且最终接受/merge 的版本可成为 SFT target；
- 在相同 prompt/context 下未被选择的真实候选成为 preference rejected；
- 同一决策因此可以同时 `sft.eligible=true` 和 `dpo.eligible=true`。

若作者选择“都不要”，保留 A/B 的 rejection。继续产生 C 时，只有作者实际接受 C 后才能作为 SFT target；可记录 C 相对 A/B 的真实偏好，但导出训练集时再决定是否拆成多个 pair，避免过度加权一次决策。

若模型提出了错误问题而作者拒绝，例如忽略人物卡中的既定异能，则保存为 reviewer false-positive preference。正确候选可以是“保留原文/不要标记此处为问题”。

## Canonical record

以 `data/schema.json`（JSON Schema Draft 2020-12）为唯一结构定义。所有文件均为
`{"schema_version": "3.0", "metadata": {...}, "records": [...]}`，不得再写单记录根对象或根级 `sft`/`dpo` 数组。文件级来源、场景说明和历史记录放入根级 `metadata`。

每条 `records[]` 必须包含：

- `id`：全数据集唯一且稳定，也是该条 SFT 的训练 ID。
- `prompt`：`user_request`、`context`、`original_text`、`metadata`；context 和原文只能为字符串或 null，不再写对象。任务必须显式提供，loader 不从场景或 review 推测。
- `sft`：`eligible`、`response`、`metadata`；可训练时 response 必须为非空字符串。不用于 SFT 时显式设 false，无目标文本时 response 为 null。target_type 放在其 metadata。
- `dpo`：`eligible`、`chosen_candidate_id`、`candidates`、`metadata`。候选必须包含 `candidate_id`、完整 `response`、`status`、`pair_id`、`metadata`；origin 放在候选 metadata。选择来源、理由及 prompt_source 放在 dpo.metadata。
- `metadata`：保留 `review`、`source`、`quality`、历史版本等事实，不参与 loader 的 prompt 拼接。
  `tags` 必须包含 `primary`、`secondary` 和 `rationale`；主次标签都只能使用
  `continuity`、`plot`、`spatial_logic`、`dialogue`、`cinematic`、`tension`、
  `presentation`、`pov`、`foreshadowing`、`everyday_life`。`no_change` 只写入
  `metadata.review.outcome`，不能作为标签。

无偏好对时仍保留 dpo 结构，eligible=false、chosen_candidate_id=null、candidates=[]。
有偏好对时 chosen_candidate_id 必须明确引用真实候选，每个 status=rejected 的候选必须有全数据集唯一且稳定的 pair_id；其他候选的 pair_id 为 null。
loader 将所选候选与每个 rejected 候选分别导出，SFT response 可以是后续最终修订，不得自动替代 DPO chosen。
新记录的 ID 可用 `prNN-chN-edit-name`，pair_id 可用 `<record-id>:rejected:<candidate-id>`。

## Preference 质量要求

高质量 pair 应满足：
- A/B 都是模型在真实任务中可能给出的合理响应，而非“正确答案 vs 故意垃圾”；
- 两者只比较同一编辑决策，避免一次同时改变多个变量；
- 作者选择是真实发生的，不根据 merge 结果倒推虚构另一个候选；
- 能保存作者理由时保存；作者只点 A/B 也有效，`preference_reason` 可为 null；
- 对 false positive，优先保留“no change > unnecessary edit”信号。

## 合成数据

合成记录使用相同 schema，但必须设置 `metadata.data_origin=synthetic` 并完整填写
`metadata.synthetic_provenance`。父记录、事实源、生成模型与参数、模板版本、评审信息和审核状态
必须可追溯。模型评审不得标成 `human_accepted` 或 `explicit_author_choice`。

合成 DPO 可以构造新的同 prompt 候选比较，但不得写回 `data/real`，也不得声称是历史作者选择。
只有通过质量门槛的记录才可设置训练 eligibility；未通过或平局样本保留 provenance 后排除训练。

## 导出训练集

Canonical JSON 是档案层，不直接绑定某个训练框架。训练前再投影：

- Executor SFT：`context + original_text + metadata.review.edit_instruction -> accepted revision`。
- Autonomous editor SFT：`repository context + broad user_request + chapter -> accepted corrected chapter`。
- Reviewer SFT：`repository context + broad user_request + chapter -> discovered issue`。
- Preference/DPO：同一 `prompt` 下导出 `chosen` 与 `rejected`。

不要因为某条记录同时支持多种 projection，就复制或篡改 canonical provenance。
当前 `LocalNovelDataset.py` 固定导出 `user_request + 上下文：context + 原文：original_text`，
空 context/原文不拼接；metadata 仅作档案，不自动导出其他 projection。

## 提交前检查

保存数据前确认：事实源已读；问题真实；before/after 或候选文本为真实版本；accepted/rejected 状态来自作者行为；历史数据没有人工补造 preference；每条记录只表达一个主要编辑决策；JSON 符合 `data/schema.json`。
运行 `python data/count.py` 会通过 loader 验证结构、候选引用和 ID 唯一性，并核对训练数量。
