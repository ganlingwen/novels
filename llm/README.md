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
  --per-device-batch-size 2 \
  --gradient-accumulation 8 \
  --max-steps 9000 \
  --learning-rate 1e-5 \
  --no-gradient-checkpointing \
  --causal-right-padding
```

这是**全参数 SFT**，不是 LoRA。脚本只对 assistant response 计算 loss；system/user prompt
全部 mask 为 `-100`。Qwen3 thinking 在模板中关闭，避免把小说训练成思维链输出。

训练入口会关闭 PyTorch 的可选 native JIT override，使用普通 CUDA 实现，避免自动触发
Triton 编译依赖。

脚本会强制检查 Stage-1 数据总数必须为 **72,573**，避免 Hugging Face 数据仓库以后新增
其他 agent 数据时被误混入训练。

默认先用 2048 context、9,000 optimizer steps（effective batch 16 时约等于 2 个数据集遍历）。GB10 推荐先尝试 `batch=2 × accumulation=8`；如果统一内存不足，退回 `batch=1 × accumulation=16` 并移除 `--no-gradient-checkpointing`。数据集不自行随机化；训练集每轮由 `DataLoader(shuffle=True)` 打乱，验证集保持固定。开始正式长跑前建议先跑一个短 benchmark，观察 GB10 的
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

`--causal-right-padding` 仅适用于本数据集的右侧 padding、padding labels=-100、因果 attention；
有效 token 看不到右侧 padding，因此省略 padding mask 可使用 SDPA 的高效 GQA 路径。
`python test_causal_padding.py` 检查 loss 和所有参数梯度一致性。
用 `--benchmark --max-steps 4 --output-dir /tmp/qwen-benchmark` 做短测，跳过验证与模型保存。

训练 tqdm 每个 optimizer step 显示 `loss / tok/s / sec/step / peak memory / batch×accum / lr`。TensorBoard 同步记录 `train/tokens_per_second`、`train/peak_memory_gb`、`train/batch_size`、`train/gradient_accumulation` 和 `train/effective_batch_size`，用于在 DGX Spark 上比较不同真实 batch 与 gradient accumulation 配置。
