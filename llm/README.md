# Qwen3-4B 小说 Stage-1 SFT

目标：在 Qwen3-4B 上复现小说 Stage-1 全参数 SFT。数据固定使用
`mikuhhn1239/novel-agent-sft-dataset` 的 `base-sft/continuation.jsonl` 与
`base-sft/instruction.jsonl`，合计 72,573 条 ChatML 样本。

## 安装

```bash
cd llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

## 训练

单卡 DGX Spark / GB10 起步配置：

```bash
python train_qwen3_4b_sft.py \
  --model Qwen/Qwen3-4B \
  --max-length 2048 \
  --per-device-batch-size 1 \
  --gradient-accumulation 16 \
  --max-steps 9000 \
  --learning-rate 1e-5 \
  --no-gradient-checkpointing \
  --causal-right-padding \
  --fused-adamw
```

这是**全参数 SFT**，不是 LoRA。脚本只对 assistant response 计算 loss；system/user prompt
全部 mask 为 `-100`。Qwen3 thinking 在模板中关闭，避免把小说训练成思维链输出。

训练入口会关闭 PyTorch 的可选 native JIT override，使用普通 CUDA 实现，避免自动触发
Triton 编译依赖。

脚本会强制检查 Stage-1 数据总数必须为 **72,573**，避免 Hugging Face 数据仓库以后新增
其他 agent 数据时被误混入训练。

默认先用 2048 context、9,000 optimizer steps（effective batch 16 时约等于 2 个数据集遍历）。GB10 实测 `batch=1 × accumulation=16` 关闭 checkpointing 比 batch 2、4 更快；如果统一内存不足，移除 `--no-gradient-checkpointing`。数据集不自行随机化；训练集每轮由 `DataLoader(shuffle=True)` 打乱，验证集保持固定。开始正式长跑前建议先跑一个短 benchmark，观察 GB10 的
tokens/s、显存/统一内存占用以及样本 truncation 比例，再决定是否改 4096 或 batch/accumulation。

## 断点恢复

```bash
python train_qwen3_4b_sft.py \
  --checkpoint outputs/qwen3-4b-novel-sft/checkpoint-XXXX \
  --max-steps 9000
```

## 输出

默认输出到：

```text
outputs/qwen3-4b-novel-sft/
```

包含 Transformers 可直接加载的模型、tokenizer 和 checkpoints。训练日志写入 `outputs/qwen3-4b-novel-sft/tensorboard/`，可用：

```bash
tensorboard --logdir outputs/qwen3-4b-novel-sft/tensorboard
```



## 代码结构

- `NovelSFTDataset.py`: 一次性加载 72,573 条数据，并显式切分 train/valid；`NovelSFTDataset(Dataset)` 只负责 tokenize 和 collate。
- `Train.py`: `TrainConfig` 将数据加载、优化器和训练调度参数分组；`Train` 实现显式训练循环，`train_step()` 完成一个 optimizer step，`valid_step()` 完成一次 validation pass。
- `train_qwen3_4b_sft.py`: main 入口，构造 dataset/model/trainer，加载 checkpoint 并调用 `train()`。

训练进度使用 tqdm，以 optimizer step 为单位显示 `loss / lr / sec`；默认每 500 step validation、每 1000 step checkpoint。


### 性能监控

GB10 实测（2048 context、effective batch 16、同一 seed；4 steps，排除第一个 warm-up step）：

| 配置 | 秒/optimizer step | 峰值分配 GiB |
| --- | ---: | ---: |
| batch 1 × accum 16，checkpointing | 32.3 | 37.7 |
| batch 2 × accum 8，无 checkpointing | 32.0 | 62.8 |
| 同上 + `--causal-right-padding` | 27.9 | 60.5 |
| batch 4 × accum 4，无 checkpointing + causal | 28.3 | 91.0 |
| batch 2 × accum 8，无 checkpointing + causal + `--fused-adamw` | 27.2 | 60.5 |
| batch 1 × accum 16，无 checkpointing + causal + fused | 26.3 | 45.4 |
| 推荐配置，8-step 确认，CUDA 同步计时 | 26.0 | 45.4 |

8-step 确认运行成功退出，排除 warm-up 后平均 25.97 秒/step（23.38–27.47 秒），
相对本次 32.28 秒基线约减少 19.5% 时间。9,000 steps 的纯训练估算约 64.9 小时，
另加验证与保存；与之前报告的 27 秒/step 相比改善有限，不承诺大幅缩短到十几小时。
118 次 GPU 采样利用率平均 94.2%（90–95%），没有明显空转；高利用率不等于计算峰值效率。

`--fused-adamw` 使用 PyTorch 自带 CUDA AdamW，不增加依赖。短测收益较小，可能受运行波动影响。

以上为 2026-09-18 在 GB10、PyTorch 2.13.0+cu130 上的真实训练测试，均使用 seed 42、
同一数据顺序、全参数更新。只改变 micro-batch 时，每批有效 token 数不同，loss 的分组平均也略有不同。
保持 effective batch 16 不代表与原配置逐位相同。
数据加载、前向、反向、梯度裁剪与 optimizer update 均计入 step；验证与保存另计。
避免在另一个训练进程运行时比较。更大 batch 未必更快：它增加 padding 和内存流量。

对照命令（每次使用独立输出目录）：

```bash
# 原始基线
python train_qwen3_4b_sft.py --benchmark --max-steps 4 --output-dir /tmp/qwen-baseline
# 推荐配置
python train_qwen3_4b_sft.py --benchmark --max-steps 8 \
  --per-device-batch-size 1 --gradient-accumulation 16 \
  --no-gradient-checkpointing --causal-right-padding --fused-adamw \
  --output-dir /tmp/qwen-fast
```

更换 batch 测试时同时调整 accumulation，保持乘积 16。比较 TensorBoard 中 warm-up 后的
`train/step_seconds` 和 `train/tokens_per_second`，不能只比较 GPU utilization。
测试中 batch 4 的峰值约 91 GiB，因此未继续尝试 batch 8/16，以避免统一内存耗尽。
原生 Flash SDPA 在 Qwen3 的 32 query heads / 8 KV heads / head_dim 128、2048 token
forward+backward 微测中为 3.45 ms，cuDNN 为 3.90 ms，未强制切换 backend。

`--causal-right-padding` 仅适用于本数据集的右侧 padding、padding labels=-100、因果 attention；
有效 token 看不到右侧 padding，因此省略 padding mask 可使用 SDPA 的高效 GQA 路径。
`python test_causal_padding.py` 检查 loss 和所有参数梯度一致性。
用 `--benchmark --max-steps 4 --output-dir /tmp/qwen-benchmark` 做短测，跳过验证与模型保存。

训练 tqdm 每个 optimizer step 显示 `loss / tok/s / sec/step / peak memory / batch×accum / lr`。TensorBoard 同步记录 `train/tokens_per_second`、`train/peak_memory_gb`、`train/batch_size`、`train/gradient_accumulation` 和 `train/effective_batch_size`，用于在 DGX Spark 上比较不同真实 batch 与 gradient accumulation 配置。

step 计时在边界同步 CUDA，不把上一次 validation/checkpoint 的耗时算入下一步。
token 数在 CPU batch 上统计，loss 在 GPU 累加后每个 optimizer step 读取一次；该调整实测
26.4 秒/step，与之前 26.3 秒相比没有明确性能收益，主要保证计时准确。
