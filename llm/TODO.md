# Qwen3-4B 小说微调实验清单

按顺序执行，先建立可信的自动评估，再优化训练。本文仅规划实验，不启动任务。

## 实验约定

- 当前长跑完成后再运行 GPU 实验；新环境、新输出目录，不覆盖已有 checkpoint。
- 每次只改一个主要因素；固定模型/数据版本、tokenizer、模板、seed、数据顺序和评估集。
- 当前参考配置：BF16 全参数 SFT，context 2048，batch 1 × accumulation 16，LR 1e-5，cosine 9000 steps、最低 LR 1e-6，无 warmup，无 gradient checkpointing，causal-right-padding、fused AdamW。
- 短实验仍固定 scheduler 的计划总长度，不用缩短 `max_steps` 同时改变 cosine 曲线；需要独立的提前结束机制。
- 性能实验先预热，再测至少 30 个稳定 steps，重复 3 次；记录编译时间、step 中位数/P95、有效输入 tokens/s、监督 tokens/s、峰值内存、总耗时。质量实验先筛选 300–500 steps，候选再延长并复核多个 seed。
- 默认性能入选门槛：稳定吞吐提升至少 10%，统一评估的 NLL 增幅不超过 0.02，偏好指标无明确退化。这是实验前约定的工程阈值，不是质量保证。
- 更改 batch、context 或 packing 时，同时报告已见样本数及监督 token 数，不能仅按 optimizer steps 比较。

## P0：先确认模型是否真的改善

- [x] **E01：审计 loss、日志和恢复连续性。** 找到当前 writer 实际路径（`outputs/.../tensorboard/<run>/`）并与旧 `runs/<run>/tensorboard/` 区分，按事件文件及运行时间分开曲线，避免恢复后重复 step 混合。核对有限 validation 值与 best 元数据；历史 NaN 不能直接判为显示故障。绘制每 100/500 steps 的均值、离散度及 LR。用小模型验证连续训练与保存后恢复的数据位置、optimizer、scheduler 一致。完成标准：每条曲线能对应到明确进程/代码版本，NaN 与断点行为有可复现解释。

- [ ] **E02：建立无泄漏、分类型的固定评估集。** 当前代码先拼接 continuation/instruction，再取前 1% 验证，因此验证集只覆盖 continuation。保留旧验证集作历史对照；为下一轮建立按来源/作品/相关片段分组、两类均覆盖的 train/valid/test 划分，并检查近重复。对本轮已训练样本只能报告回顾性结果，不能重新命名为未见测试集。完成标准：保存样本 ID、来源、分组和重叠审计。

- [ ] **E03（部分完成）：原始模型与各 checkpoint 自动对比。** 已完成 Base 与 step 1000–9000 的 assistant-only NLL 对比和 best step 核对，记录于 [`llm/experiments/initial_evaluation.md`](experiments/initial_evaluation.md)。仍缺：使用 Stage-1 固定 validation/test 集的正式对比、PPL、continuation/instruction 分组和 bootstrap 差值区间。单个初始 batch 的 loss 不充当原始模型基线。

- [ ] **E04（核心指标完成）：手工 SFT/DPO 数据作为自动编辑基准。** 已完成 eligible 手工数据的 accepted-response NLL、DPO accuracy、长度归一化 margin，以及 Base/step 1000–9000 对比，记录于 [`llm/experiments/initial_evaluation.md`](experiments/initial_evaluation.md)。仍缺：按场景/PR 去重计权、缺失上下文审计和 bootstrap/不确定性区间；当前结果只能作为回顾性编辑基准，不能当作未见泛化测试。

- [ ] **E05：自动生成回归与通用能力保留。** 固定小说编辑、续写及少量非小说指令任务；统一模板、thinking 设置、生成长度和解码参数。对可确定的要求统计姓名/数字保留、格式合规、禁止内容、重复 n-gram、意外截断及无须修改样本的误改率；复杂空间/剧情错误不以关键词检查冒充可靠判断。完成标准：与原始模型比较并列出检测器覆盖范围；不把自动分数直接等同文学质量。

## P1：训练质量和数值稳定性

- [ ] **E06：warmup 消融。** 从相同原始权重启动，无 warmup vs 100 steps 线性 warmup；必要时再试 300。峰值 LR、总训练预算、cosine 终点相同。比较早期梯度范数、非有限值、固定评估 NLL 和最终指标；不在已训练数千步的 checkpoint 上重新 warmup 来替代这个实验。

- [ ] **E07：学习率和训练时长。** 固定 E06 选出的 schedule，比较峰值 LR 5e-6 / 1e-5 / 2e-5；按相同监督 token 预算评估。建立 checkpoint 质量随训练时长变化表，判断平台期/过拟合；用 valid 选停止点，test 只用于最终确认。完成标准：收益超过评估波动，且无通用能力明显回退。

- [ ] **E08：优化器精度审计。** 检查实际参数、梯度及 AdamW 一二阶状态 dtype，确认是否存在 FP32 主权重，不能将 autocast 自动等同混合精度优化器。若更新/状态精度可疑，比较现状与显式 FP32 状态/主权重方案，记录更新为零的比例、更新/权重范数、NLL、内存和速度；先估算内存再运行。

- [ ] **E09：监督长度、截断与 loss 加权。** 统计两类数据 prompt/response 长度、截断比例和保留监督 token 比例。分别比较当前样本均值目标与整个 accumulation 窗口的监督 token 加权目标；另做 context 2048 vs 4096 实验。长上下文评估同时报告共同截断口径和完整上下文口径，防止截断差异误导结论。

## P2：DGX Spark 单卡性能

- [ ] **E10：BF16 eager vs `torch.compile`。** 先建立当前环境下可重复基线，再验证编译、动态长度和 recompilation 开销。当前入口禁用 native JIT override，编译实验需单独确认配置兼容性。完成标准：短期数值一致性、稳定吞吐及包含编译成本的回本时间。

- [ ] **E11：FP8 训练（优先于 FP4）。** 在独立环境核实 GB10、ARM64、CUDA/PyTorch/torchao 与 kernel 的实际兼容性；先小模型前反向，再跑 Qwen3-4B。比较 BF16 eager、BF16 compile、FP8 compile，分离编译收益与精度收益；先试 rowwise，再按需要试 tensorwise/高精度权重梯度 recipe。明确转换的 Linear 层及保留高精度的部分，不把 optimizer 全部改为 FP8。检查非有限 loss/梯度、保存恢复和 BF16 导出评估；先跑 100–300 steps，通过后扩大质量验证。完成标准：达到公共性能/质量门槛。官方多卡加速数字不作为 Spark 预期值。[torchao 官方训练说明](https://docs.pytorch.org/ao/stable/workflows/training.html)

- [ ] **E12：融合 kernel。** 独立测试 Liger 的适用模块，尤其 Linear + cross-entropy，先确认 Qwen3、assistant mask、GB10/ARM64 与当前软件版本兼容。使用同一小批次核对 loss/梯度，再测全参数训练吞吐、内存；明确是否与 compile/FP8 组合兼容。[Liger 官方仓库](https://github.com/linkedin/Liger-Kernel)

- [ ] **E13：数据供给、batch 与 packing。** 先 profile 判断 tokenization/等待占比，比较缓存 tokenization 和 workers 0/2/4。已有 README 中 batch 1×16 优于 2×8/4×4 的结果，只有 kernel/精度改变后才重新测试。当前 batch 1 本就没有跨样本 padding；packing 的潜在收益在短序列利用率。若做 packing，必须隔离样本 attention、正确处理 position/labels，验证不跨样本泄漏；不能直接沿用无边界的 causal mask。

- [ ] **E14：框架对照。** 在现有循环、TRL/支持 Qwen3 的 Unsloth 路径中选一个候选，先验证 Spark/ARM64 安装与全参数训练支持，再保持数据、模板、loss mask、有效 batch、LR 和精度一致比较。LoRA/QLoRA 的结果单列为 E15，不当作全参数框架加速。只在可重复收益抵消迁移成本时采用；单卡不为分布式功能引入 FSDP/DeepSpeed/Megatron。[Unsloth 官方仓库](https://github.com/unslothai/unsloth)

## P3：后续方法探索

- [ ] **E15：LoRA/QLoRA 与全参数 SFT。** 同一初始模型、训练数据和评估，先比较 LoRA rank 16/64，再按资源需要考虑 QLoRA。分别报告同 token 预算和同墙钟预算的质量、内存、吞吐；4-bit 冻结底座的 QLoRA 不等于 FP4 全参数训练。

- [ ] **E16：手工数据二阶段 SFT → DPO。** 先冻结 E04 的独立测试组，其余 accepted 样本做低 LR 的二阶段 SFT，再以真实同 prompt 偏好对比较“只 SFT”与“SFT+DPO”。与未进行二阶段训练的模型一起测偏好准确率、编辑 NLL 和 E05 回归。样本不足以分组留出时继续采集，先不训练；训练后不再把同批样本当泛化证据。

- [ ] **E17：FP4 可行性（低优先级）。** FP8 验证完成后，再核实当前 GB10 软件栈是否提供真正低精度前反向训练；分别识别 NVFP4/MXFP4、推理量化、QAT 模拟量化、QLoRA，不能混称 FP4 训练。没有兼容 kernel 则记录不适用；有支持则从小模型验证数值、恢复和导出，再按 E11 对照，不用硬件峰值推算端到端收益。

## 每项实验结果模板

```text
实验 ID / 日期 / 结论：采用、放弃或证据不足
代码 commit / 环境及依赖版本 / GPU
模型及数据 revision / 评估集 ID / seed
唯一主要变量 / 基线 / 参数 / 实际命令
起始 checkpoint / optimizer 是否恢复 / scheduler 总长度
已见样本与监督 tokens / 总耗时 / 编译耗时
step 中位数与 P95 / tokens/s / 峰值内存
分类型 NLL/PPL / 偏好准确率与 margin / 回归指标 / 不确定性
日志与 checkpoint 路径 / 异常 / 下一步
```
