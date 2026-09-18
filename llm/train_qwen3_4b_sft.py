#!/usr/bin/env python3
import argparse
from itertools import chain

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Full-parameter Stage-1 SFT for Qwen3-4B on the 72,573-example novel dataset.")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="mikuhhn1239/novel-agent-sft-dataset")
    p.add_argument("--output-dir", default="outputs/qwen3-4b-novel-sft")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=9000)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args()


def load_novel_data(repo_id):
    # Pin the two Stage-1 files explicitly. This avoids accidentally training on
    # the repository's additional agent datasets.
    files = {
        "continuation": "base-sft/continuation.jsonl",
        "instruction": "base-sft/instruction.jsonl",
    }
    parts = []
    for name, path in files.items():
        ds = load_dataset(repo_id, data_files={"train": path}, split="train")
        print(f"{name}: {len(ds):,}")
        parts.append(ds)
    ds = concatenate_datasets(parts).shuffle(seed=42)
    print(f"total: {len(ds):,}")
    if len(ds) != 72573:
        raise RuntimeError(f"Expected 72,573 Stage-1 rows, got {len(ds):,}. Dataset layout may have changed.")
    return ds


def encode_example(example, tokenizer, max_length):
    messages = example["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected ChatML messages ending with assistant.")

    # Prompt tokens are masked. Only the assistant answer contributes to loss.
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    # Keep the beginning of the prompt and assistant response under max_length.
    # Stage-1 samples are expected to fit 2K; truncation is intentionally simple
    # and counted below so it is visible rather than silent.
    full_ids = full_ids[:max_length]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "was_truncated": len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)) > max_length,
    }


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            n = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [pad] * n)
            batch["attention_mask"].append(x["attention_mask"] + [0] * n)
            batch["labels"].append(x["labels"] + [-100] * n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_novel_data(args.dataset)
    train = raw.map(
        lambda x: encode_example(x, tokenizer, args.max_length),
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )
    truncated = sum(train["was_truncated"])
    print(f"truncated: {truncated:,}/{len(train):,} ({truncated / len(train):.2%})")
    train = train.remove_columns(["was_truncated"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        bf16=True,
        tf32=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
        save_safetensors=True,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/tensorboard",
        run_name="qwen3-4b-novel-stage1-sft",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        data_collator=CausalLMCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
