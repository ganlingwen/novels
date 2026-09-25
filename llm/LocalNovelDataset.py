"""Validated canonical SFT and DPO data loader for Stage-2 training."""

import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from jsonschema import Draft202012Validator, ValidationError
from torch.utils.data import Dataset as TorchDataset

from NovelSFTDataset import NovelSFTDataset


def _prompt_text(prompt: dict) -> str:
    parts = [prompt["user_request"]]
    if prompt["context"]:
        parts.append(f"上下文：{prompt['context']}")
    if prompt["original_text"]:
        parts.append(f"原文：{prompt['original_text']}")
    return "\n\n".join(parts)


def _record_paths(data_dir: str | Path) -> list[Path]:
    """Find real records under data/real, accepting either data or data/real."""
    root = Path(data_dir)
    record_dir = root if root.name == "real" else root / "real"
    if not record_dir.is_dir():
        raise FileNotFoundError(f"Real training data directory not found: {record_dir}")
    paths = sorted(record_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No real training records found in {record_dir}")
    return paths


def _local_records(data_dir: str | Path):
    """Validate schema and cross-field invariants; never guess or silently drop data."""
    schema_path = Path(__file__).resolve().parents[1] / "data" / "schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    record_ids, pair_ids = set(), set()
    for path in _record_paths(data_dir):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
            validator.validate(bundle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error}") from error
        except ValidationError as error:
            raise ValueError(f"{path}: {error.json_path}: {error.message}") from error
        for item in bundle["records"]:
            location = f"{path}: record {item['id']!r}"
            if item["id"] in record_ids:
                raise ValueError(f"{location}: duplicate record ID")
            record_ids.add(item["id"])
            dpo = item["dpo"]
            candidates = {c["candidate_id"]: c for c in dpo["candidates"]}
            if len(candidates) != len(dpo["candidates"]):
                raise ValueError(f"{location}: duplicate candidate ID")
            chosen_id = dpo["chosen_candidate_id"]
            if chosen_id is not None and chosen_id not in candidates:
                raise ValueError(f"{location}: chosen_candidate_id does not reference a candidate")
            for candidate in candidates.values():
                pair_id = candidate["pair_id"]
                if dpo["eligible"] and candidate["status"] == "rejected":
                    if candidate["response"] == candidates[chosen_id]["response"]:
                        raise ValueError(f"{location}: chosen and rejected responses must differ")
                    if pair_id is None or pair_id in pair_ids:
                        raise ValueError(f"{location}: missing or duplicate pair ID")
                    pair_ids.add(pair_id)
                elif pair_id is not None:
                    raise ValueError(f"{location}: pair_id is only valid for an eligible rejected candidate")
            yield path, item


def load_local_sft_records(data_dir: str | Path) -> list[dict]:
    """Project explicitly eligible canonical records into SFT samples."""
    return [
        {
            "id": item["id"],
            "source": path.name,
            "prompt": _prompt_text(item["prompt"]),
            "response": item["sft"]["response"],
        }
        for path, item in _local_records(data_dir)
        if item["sft"]["eligible"]
    ]


def split_local_sft(records: list[dict], validation_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    ordered = sorted(records, key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest())
    n_valid = max(1, round(len(ordered) * validation_ratio))
    return ordered[n_valid:], ordered[:n_valid]


def make_sft_dataset(records: list[dict], tokenizer, max_length: int) -> NovelSFTDataset:
    rows = [
        {
            "messages": [
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": record["response"]},
            ]
        }
        for record in records
    ]
    return NovelSFTDataset(Dataset.from_list(rows), tokenizer, max_length)


def load_local_dpo_records(data_dir: str | Path) -> list[dict]:
    """Pair the explicit chosen candidate with each eligible rejected candidate."""
    records = []
    for path, item in _local_records(data_dir):
        dpo = item["dpo"]
        if not dpo["eligible"]:
            continue
        candidates = {c["candidate_id"]: c for c in dpo["candidates"]}
        chosen = candidates[dpo["chosen_candidate_id"]]["response"]
        for candidate in candidates.values():
            if candidate["status"] == "rejected":
                records.append(
                    {
                        "id": candidate["pair_id"],
                        "source": path.name,
                        "prompt": _prompt_text(item["prompt"]),
                        "chosen": chosen,
                        "rejected": candidate["response"],
                    }
                )
    return records


def split_local_dpo(records: list[dict], validation_ratio: float = 0.1) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    ordered = sorted(records, key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest())
    n_valid = max(1, round(len(ordered) * validation_ratio))
    return ordered[n_valid:], ordered[:n_valid]


class NovelDPODataset(TorchDataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 2048):
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        prompt_messages = [{"role": "user", "content": record["prompt"]}]
        full_prompt_ids = self._tokenize(prompt_messages, add_generation_prompt=True)
        prompt_ids = full_prompt_ids[-min(len(full_prompt_ids), self.max_length // 2) :]
        result = {"prompt_ids": torch.tensor(prompt_ids, dtype=torch.long)}
        for name in ("chosen", "rejected"):
            full_ids = self._tokenize(
                prompt_messages + [{"role": "assistant", "content": record[name]}], add_generation_prompt=False
            )
            if full_ids[: len(full_prompt_ids)] != full_prompt_ids:
                raise ValueError("Expected the generation prompt to prefix the full conversation.")
            response_ids = full_ids[len(full_prompt_ids) :]
            response_ids = response_ids[: self.max_length - len(prompt_ids)]
            if not response_ids:
                raise ValueError("Expected a non-empty preference response.")
            result[f"{name}_ids"] = torch.tensor(response_ids, dtype=torch.long)
        return result

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
        pad = self.tokenizer.pad_token_id
        max_prompt = max(item["prompt_ids"].numel() for item in items)
        batch = {}
        batch["prompt_ids"] = torch.stack(
            [
                torch.nn.functional.pad(item["prompt_ids"], (0, max_prompt - item["prompt_ids"].numel()), value=pad)
                for item in items
            ]
        )
        batch["prompt_mask"] = batch["prompt_ids"].ne(pad)
        for name in ("chosen", "rejected"):
            max_response = max(item[f"{name}_ids"].numel() for item in items)
            batch[f"{name}_ids"] = torch.stack(
                [
                    torch.nn.functional.pad(
                        item[f"{name}_ids"], (0, max_response - item[f"{name}_ids"].numel()), value=pad
                    )
                    for item in items
                ]
            )
            batch[f"{name}_mask"] = batch[f"{name}_ids"].ne(pad)
        return batch
