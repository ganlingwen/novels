#!/usr/bin/env python3
import argparse
import os

# PyTorch 2.13 may otherwise route a Qwen3 RoPE bmm through an optional
# Triton native override. The regular CUDA implementation is sufficient here.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from NovelSFTDataset import NovelSFTDataset, load_stage1_dataset, split_stage1_dataset
from Train import DataLoaderConfig, OptimizerConfig, Train, TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="mikuhhn1239/novel-agent-sft-dataset")
    p.add_argument("--output-dir", default="outputs/qwen3-4b-novel-sft")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=9000)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--valid-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="Run max-steps without validation or saving model/checkpoints.")
    p.add_argument("--causal-right-padding", action="store_true", help="Use causal SDPA without a padding mask; requires right padding and ignored padding labels.")
    p.add_argument("--validation-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-parameter Qwen3-4B SFT, but no CUDA device is available.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = load_stage1_dataset(args.dataset)
    train_raw, valid_raw = split_stage1_dataset(raw_dataset, args.validation_ratio)
    train_dataset = NovelSFTDataset(train_raw, tokenizer, args.max_length)
    valid_dataset = NovelSFTDataset(valid_raw, tokenizer, args.max_length)
    print(f"train: {len(train_dataset):,}; valid: {len(valid_dataset):,}")
    model_path = args.checkpoint or args.model
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainer_config = TrainConfig(
        causal_right_padding=args.causal_right_padding,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        gradient_accumulation=args.gradient_accumulation,
        valid_steps=0 if args.benchmark else args.valid_steps,
        save_steps=0 if args.benchmark else args.save_steps,
        data_loader=DataLoaderConfig(batch_size=args.per_device_batch_size, num_workers=args.num_workers),
        optimizer=OptimizerConfig(learning_rate=args.learning_rate),
    )
    trainer = Train(model, train_dataset, valid_dataset, trainer_config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    if args.benchmark:
        return
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
