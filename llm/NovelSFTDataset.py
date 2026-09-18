import random

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


class NovelSFTDataset(Dataset):
    FILES = {
        "continuation": "base-sft/continuation.jsonl",
        "instruction": "base-sft/instruction.jsonl",
    }
    EXPECTED_SIZE = 72573

    def __init__(self, repo_id, tokenizer, max_length=2048, seed=42, validation_ratio=0.01, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        parts = []
        for name, path in self.FILES.items():
            ds = load_dataset(repo_id, data_files={"train": path}, split="train")
            print(f"{name}: {len(ds):,}")
            parts.append(ds)
        raw = concatenate_datasets(parts)
        if len(raw) != self.EXPECTED_SIZE:
            raise RuntimeError(f"Expected {self.EXPECTED_SIZE:,} rows, got {len(raw):,}.")
        indices = list(range(len(raw)))
        random.Random(seed).shuffle(indices)
        n_valid = max(1, int(len(indices) * validation_ratio))
        indices = indices[-n_valid:] if split == "valid" else indices[:-n_valid]
        self.raw = raw.select(indices)
        print(f"{split}: {len(self.raw):,}")

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        messages = self.raw[index]["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Expected messages ending with assistant.")
        prompt_ids = self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor([-100] * prompt_len + full_ids[prompt_len:], dtype=torch.long),
        }

    def collate_fn(self, items):
        max_len = max(x["input_ids"].numel() for x in items)
        pad = self.tokenizer.pad_token_id
        batch = {}
        batch["input_ids"] = torch.stack([torch.nn.functional.pad(x["input_ids"], (0, max_len - x["input_ids"].numel()), value=pad) for x in items])
        batch["attention_mask"] = torch.stack([torch.nn.functional.pad(x["attention_mask"], (0, max_len - x["attention_mask"].numel()), value=0) for x in items])
        batch["labels"] = torch.stack([torch.nn.functional.pad(x["labels"], (0, max_len - x["labels"].numel()), value=-100) for x in items])
        return batch
