#!/usr/bin/env python3
"""Compare Stage-2 checkpoints on the same local DPO benchmark."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from DPOTrainer import sequence_logprob
from LocalNovelDataset import NovelDPODataset, load_local_dpo_records, split_local_dpo


@torch.no_grad()
def evaluate_model(model, tokenizer, records, valid_records, max_length):
    def score(dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
        chosen_nll = rejected_nll = 0.0
        chosen_tokens = rejected_tokens = 0
        normalized_correct = 0
        normalized_margin = 0.0
        raw_correct = 0
        total = 0
        model.eval()
        for batch in loader:
            chosen_len = batch["chosen_mask"].sum(dim=1).cuda()
            rejected_len = batch["rejected_mask"].sum(dim=1).cuda()
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            chosen = sequence_logprob(model, batch, "chosen")
            rejected = sequence_logprob(model, batch, "rejected")
            chosen_nll -= chosen.item()
            rejected_nll -= rejected.item()
            chosen_tokens += int(chosen_len.item())
            rejected_tokens += int(rejected_len.item())
            margin = chosen / chosen_len - rejected / rejected_len
            normalized_margin += margin.item()
            normalized_correct += int((margin > 0).item())
            raw_correct += int((chosen > rejected).item())
            total += 1
        return {
            "records": total,
            "chosen_nll": chosen_nll / chosen_tokens,
            "rejected_nll": rejected_nll / rejected_tokens,
            "normalized_accuracy": normalized_correct / total,
            "normalized_margin": normalized_margin / total,
            "raw_accuracy": raw_correct / total,
        }

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return {
        "all": score(NovelDPODataset(records, tokenizer, max_length)),
        "valid": score(NovelDPODataset(valid_records, tokenizer, max_length)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("models", nargs="+", help="NAME=MODEL_PATH")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    records = load_local_dpo_records(args.data_dir)
    _, valid_records = split_local_dpo(records, 0.1)
    result = {"records": len(records), "valid_records": len(valid_records), "models": {}}
    for item in args.models:
        name, path = item.split("=", 1)
        print(f"evaluating {name}")
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        result["models"][name] = evaluate_model(model, tokenizer, records, valid_records, args.max_length)
        del model
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
