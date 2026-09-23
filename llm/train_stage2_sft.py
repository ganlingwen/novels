#!/usr/bin/env python3
"""Low-learning-rate Stage-2 SFT on local editorial records."""

import argparse
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from LocalNovelDataset import load_local_sft_records, make_sft_dataset, split_local_sft
from SFTTrainer import DataLoaderConfig, OptimizerConfig, SFTTrainer, SFTTrainerConfig, create_run_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-model", required=True, help="Stage-1 final/best model directory; optimizer state is not restored."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/qwen3-4b-novel-stage2-sft")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--causal-right-padding", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-2 SFT")
    set_seed(args.seed)
    run_dir = create_run_directory(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_local_sft_records(args.data_dir)
    train_records, valid_records = split_local_sft(records, args.validation_ratio)
    train_dataset = make_sft_dataset(train_records, tokenizer, args.max_length)
    valid_dataset = make_sft_dataset(valid_records, tokenizer, args.max_length)
    print(f"run directory: {run_dir}")
    print(f"local SFT: {len(records)}; train: {len(train_dataset)}; valid: {len(valid_dataset)}")
    model = AutoModelForCausalLM.from_pretrained(
        args.init_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    config = SFTTrainerConfig(
        causal_right_padding=args.causal_right_padding,
        output_dir=str(run_dir),
        max_steps=args.max_steps,
        gradient_accumulation=args.gradient_accumulation,
        valid_steps=0,
        save_steps=0,
        save_best=False,
        seed=args.seed,
        data_loader=DataLoaderConfig(batch_size=args.per_device_batch_size, num_workers=args.num_workers),
        optimizer=OptimizerConfig(learning_rate=args.learning_rate, fused=args.fused_adamw),
        lr_schedule=args.lr_schedule,
    )
    trainer = SFTTrainer(model, train_dataset, valid_dataset, config)
    trainer.train()
    final_validation_loss = trainer.valid_step()
    print(f"final validation loss: {final_validation_loss:.6f}")
    final_dir = run_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
