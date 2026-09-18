# 编辑训练数据沉淀规范

本 skill 用于把真实小说编辑过程沉淀为可训练的 SFT 与 preference/DPO 数据。目标不是为了凑数据而制造差异，而是忠实保存“模型发现问题 → 提案 → 作者选择/拒绝 → 合并”的真实决策。

## 触发时机

当正文 review、局部修改或 PR/MR 被作者接受、拒绝，或作者要求积累训练数据时使用。正文审查本身仍遵守 `AGENTS.md` 的任务路由，先读取 `skills/write.md` 及相关事实源；本 skill 只规定如何产生、选择和保存训练记录。

## 核心原则

1. 一个独立编辑决策对应一个 `data/*.json` 文件，不把互不相关的问题塞进同一记录。
2. 只把作者真实接受/merge 的修改作为 SFT target；不得把模型自评“正确”当作 human accepted。
3. Preference/DPO 必须来自同一个 prompt、同一份上下文下的真实候选比较。不得为历史 accepted edit 事后编造 rejected candidate。
4. 候选版本必须在作者选择前原样保存；chosen/rejected 都保留完整文本。
5. “不修改/保留原文”是合法候选，用于训练编辑克制和减少 false positive。
6. 详细 `edit_instruction` 若由 AI review 产生，必须标记 `review.provenance=ai_discovered_during_review`，不得伪装成作者 prompt。
7. 作者只说“继续”“复审”“改下一处”等宽泛要求时，`prompt.user_request` 应忠实表达这种自主审查任务；详细问题放在 `review`，不要塞进 user prompt。
8. 保留 repo、文件、PR、commit 等 provenance，使训练记录可追溯。

## 推荐编辑流程

每次先按事实源独立判断是否真的存在问题。若不存在值得修改的问题，应明确选择 no-change，不为产生数据而硬改。

发现一个真实问题后，优先给出两个都合理、但编辑取向有真实差异的候选 A/B。例如 A 做最小修改，B 可适度重组表达。不要故意写一个明显差的 B。

作者选择 A 或 B 后：
- chosen 且最终接受/merge 的版本可成为 SFT target；
- 在相同 prompt/context 下未被选择的真实候选成为 preference rejected；
- 同一决策因此可以同时 `sft.eligible=true` 和 `preference.eligible=true`。

若作者选择“都不要”，保留 A/B 的 rejection。继续产生 C 时，只有作者实际接受 C 后才能作为 SFT target；可记录 C 相对 A/B 的真实偏好，但导出训练集时再决定是否拆成多个 pair，避免过度加权一次决策。

若模型提出了错误问题而作者拒绝，例如忽略人物卡中的既定异能，则保存为 reviewer false-positive preference。正确候选可以是“保留原文/不要标记此处为问题”。

## Canonical record

以 `data/schema.json` 为结构定义。核心字段：

- `prompt`：作者真实任务、必要 context、原文。
- `review`：AI 发现的问题、证据、详细修改指令、原因及 provenance。
- `sft`：是否可用于 SFT，以及作者最终接受的 response。
- `preference`：同 prompt 的真实 candidates、chosen id、选择来源和可选理由。
- `source`：仓库、文件、PR 与 commit provenance。
- `quality`：human accepted/rejected、merged 等事实标签。

## Preference 质量要求

高质量 pair 应满足：
- A/B 都是模型在真实任务中可能给出的合理响应，而非“正确答案 vs 故意垃圾”；
- 两者只比较同一编辑决策，避免一次同时改变多个变量；
- 作者选择是真实发生的，不根据 merge 结果倒推虚构另一个候选；
- 能保存作者理由时保存；作者只点 A/B 也有效，`preference_reason` 可为 null；
- 对 false positive，优先保留“no change > unnecessary edit”信号。

## 导出训练集

Canonical JSON 是档案层，不直接绑定某个训练框架。训练前再投影：

- Executor SFT：`context + original_text + edit_instruction -> accepted revision`。
- Autonomous editor SFT：`repository context + broad user_request + chapter -> accepted corrected chapter`。
- Reviewer SFT：`repository context + broad user_request + chapter -> discovered issue`。
- Preference/DPO：同一 `prompt` 下导出 `chosen` 与 `rejected`。

不要因为某条记录同时支持多种 projection，就复制或篡改 canonical provenance。

## 提交前检查

保存数据前确认：事实源已读；问题真实；before/after 或候选文本为真实版本；accepted/rejected 状态来自作者行为；历史数据没有人工补造 preference；每条记录只表达一个主要编辑决策；JSON 符合 `data/schema.json`。
