#!/usr/bin/env python3
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from NovelSFTDataset import NovelSFTDataset
from Train import Train


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
    p.add_argument("--validation-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = NovelSFTDataset(args.dataset, tokenizer, args.max_length, args.seed, args.validation_ratio, "train")
    valid_dataset = NovelSFTDataset(args.dataset, tokenizer, args.max_length, args.seed, args.validation_ratio, "valid")
    model_path = args.checkpoint or args.model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    trainer = Train(model, train_dataset, valid_dataset, args.output_dir, args.max_steps, args.learning_rate, args.per_device_batch_size, args.gradient_accumulation, args.valid_steps, args.save_steps)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
