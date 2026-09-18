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
huggingface-cli login
```

## 训练

单卡 DGX Spark / GB10 起步配置：

```bash
python train_qwen3_4b_sft.py \
  --model Qwen/Qwen3-4B \
  --max-length 2048 \
  --per-device-batch-size 1 \
  --gradient-accumulation 16 \
  --epochs 2 \
  --learning-rate 1e-5
```

这是**全参数 SFT**，不是 LoRA。脚本只对 assistant response 计算 loss；system/user prompt
全部 mask 为 `-100`。Qwen3 thinking 在模板中关闭，避免把小说训练成思维链输出。

脚本会强制检查 Stage-1 数据总数必须为 **72,573**，避免 Hugging Face 数据仓库以后新增
其他 agent 数据时被误混入训练。

默认先用 2048 context、9,000 optimizer steps（effective batch 16 时约等于原先的 2 epochs）。开始正式长跑前建议先跑一个短 benchmark，观察 GB10 的
tokens/s、显存/统一内存占用以及样本 truncation 比例，再决定是否改 4096 或 batch/accumulation。

## 断点恢复

```bash
python train_qwen3_4b_sft.py --resume-from-checkpoint outputs/qwen3-4b-novel-sft/checkpoint-XXXX
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

- `NovelSFTDataset.py`: `NovelSFTDataset(Dataset)`，负责 72,573 条数据加载、train/valid 切分、tokenize 和 collate。
- `Train.py`: `Train`，显式训练循环；`train_step()` 完成一个 optimizer step，`valid_step()` 完成一次 validation pass；`train()` 只负责调度 step、TensorBoard、checkpoint 与 tqdm。
- `train_qwen3_4b_sft.py`: main 入口，构造 dataset/model/trainer，加载 checkpoint 并调用 `train()`。

训练进度使用 tqdm，以 optimizer step 为单位显示 `loss / lr / sec`；默认每 500 step validation、每 1000 step checkpoint。


### 性能监控

训练 tqdm 每个 optimizer step 显示 `loss / tok/s / sec/step / peak memory / batch×accum / lr`。TensorBoard 同步记录 `train/tokens_per_second`、`train/peak_memory_gb`、`train/batch_size`、`train/gradient_accumulation` 和 `train/effective_batch_size`，用于在 DGX Spark 上比较不同真实 batch 与 gradient accumulation 配置。
