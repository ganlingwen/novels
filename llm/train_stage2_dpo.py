#!/usr/bin/env python3
"""Stage-2.2 DPO on real same-prompt editorial preferences."""

import argparse
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from LocalNovelDataset import NovelDPODataset, load_local_dpo_records, split_local_dpo
from Train import DataLoaderConfig, create_run_directory, tensorboard_directory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-model", required=True, help="Policy model to continue from.")
    parser.add_argument("--reference-model", help="Frozen DPO reference; defaults to --init-model.")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default="outputs/qwen3-4b-novel-stage2-dpo")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--fused-adamw", action="store_true")
    return parser.parse_args()


def sequence_logprob(model, batch, choice):
    prompt_ids = batch["prompt_ids"]
    prompt_mask = batch["prompt_mask"]
    response_ids = batch[f"{choice}_ids"]
    response_mask = batch[f"{choice}_mask"]
    values = []
    for row in range(prompt_ids.size(0)):
        prompt_len = int(prompt_mask[row].sum())
        response_len = int(response_mask[row].sum())
        input_ids = torch.cat((prompt_ids[row, :prompt_len], response_ids[row, :response_len])).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids).logits[:, :-1].float()
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        response_start = prompt_len - 1
        response_end = response_start + response_len
        values.append(
            log_probs[:, response_start:response_end]
            .gather(-1, targets[:, response_start:response_end, None])
            .squeeze(-1)
            .sum()
        )
    return torch.stack(values)


@torch.no_grad()
def evaluate(policy, reference, loader, beta):
    policy.eval()
    reference.eval()
    total_loss = 0.0
    total_margin = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        policy_chosen = sequence_logprob(policy, batch, "chosen")
        policy_rejected = sequence_logprob(policy, batch, "rejected")
        reference_chosen = sequence_logprob(reference, batch, "chosen")
        reference_rejected = sequence_logprob(reference, batch, "rejected")
        margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        total_loss += (-F.logsigmoid(beta * margin)).sum().item()
        total_margin += margin.sum().item()
        total_correct += int((policy_chosen > policy_rejected).sum())
        total += len(margin)
    return total_loss / total, total_correct / total, total_margin / total


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage-2 DPO")
    if args.max_steps < 1 or args.gradient_accumulation < 1 or args.beta <= 0:
        raise ValueError("max_steps, gradient_accumulation and beta must be positive")
    set_seed(args.seed)
    reference_path = args.reference_model or args.init_model
    run_dir = create_run_directory(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_local_dpo_records(args.data_dir)
    train_records, valid_records = split_local_dpo(records, args.validation_ratio)
    train_dataset = NovelDPODataset(train_records, tokenizer, args.max_length)
    valid_dataset = NovelDPODataset(valid_records, tokenizer, args.max_length)
    print(f"run directory: {run_dir}")
    print(f"local DPO: {len(records)}; train: {len(train_dataset)}; valid: {len(valid_dataset)}")

    policy = AutoModelForCausalLM.from_pretrained(
        args.init_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    reference = (
        AutoModelForCausalLM.from_pretrained(reference_path, dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda()
        .eval()
    )
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy.config.use_cache = False
    reference.config.use_cache = False
    if not args.no_gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    loader_config = DataLoaderConfig(batch_size=1, num_workers=args.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config.batch_size,
        shuffle=True,
        num_workers=loader_config.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=loader_config.num_workers,
        collate_fn=valid_dataset.collate_fn,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.1, fused=args.fused_adamw)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 0.1
    )
    writer = SummaryWriter(str(tensorboard_directory(run_dir)))
    iterator = iter(train_loader)
    progress = tqdm(range(args.max_steps), unit="step", desc="dpo")
    for step in progress:
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.no_grad():
                reference_chosen = sequence_logprob(reference, batch, "chosen")
                reference_rejected = sequence_logprob(reference, batch, "rejected")
            policy_chosen = sequence_logprob(policy, batch, "chosen")
            policy_rejected = sequence_logprob(policy, batch, "rejected")
            margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
            loss = (-F.logsigmoid(args.beta * margin)).mean() / args.gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss at step {step + 1}.")
            loss.backward()
            loss_value += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step = step + 1
        writer.add_scalar("train/loss", loss_value, global_step)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    final_loss, final_accuracy, final_margin = evaluate(policy, reference, valid_loader, args.beta)
    print(f"final validation dpo loss: {final_loss:.6f}")
    print(f"final validation preference accuracy: {final_accuracy:.6f}")
    print(f"final validation margin: {final_margin:.6f}")
    final_dir = run_dir / "final"
    policy.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    writer.close()


if __name__ == "__main__":
    main()
