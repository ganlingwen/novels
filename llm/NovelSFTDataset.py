import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset


FILES = {
    "continuation": "base-sft/continuation.jsonl",
    "instruction": "base-sft/instruction.jsonl",
}
EXPECTED_SIZE = 72573


def load_stage1_dataset(repo_id):
    parts = []
    for name, path in FILES.items():
        ds = load_dataset(repo_id, data_files={"train": path}, split="train")
        print(f"{name}: {len(ds):,}")
        parts.append(ds)
    raw = concatenate_datasets(parts)
    if len(raw) != EXPECTED_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_SIZE:,} rows, got {len(raw):,}.")
    return raw


def split_stage1_dataset(raw, validation_ratio=0.01):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    n_valid = max(1, int(len(raw) * validation_ratio))
    return raw.select(range(n_valid, len(raw))), raw.select(range(n_valid))


class NovelSFTDataset(Dataset):
    def __init__(self, raw, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw = raw

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        messages = self.raw[index]["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Expected messages ending with assistant.")
        prompt_ids = self._tokenize(messages[:-1], add_generation_prompt=True)
        full_ids = self._tokenize(messages, add_generation_prompt=False)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor([-100] * prompt_len + full_ids[prompt_len:], dtype=torch.long),
        }

    def _tokenize(self, messages, add_generation_prompt):
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        return list(encoded)

    def collate_fn(self, items):
        max_len = max(x["input_ids"].numel() for x in items)
        pad = self.tokenizer.pad_token_id
        batch = {}
        batch["input_ids"] = torch.stack([torch.nn.functional.pad(x["input_ids"], (0, max_len - x["input_ids"].numel()), value=pad) for x in items])
        batch["attention_mask"] = torch.stack([torch.nn.functional.pad(x["attention_mask"], (0, max_len - x["attention_mask"].numel()), value=0) for x in items])
        batch["labels"] = torch.stack([torch.nn.functional.pad(x["labels"], (0, max_len - x["labels"].numel()), value=-100) for x in items])
        return batch
