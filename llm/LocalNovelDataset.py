"""Validated canonical SFT and DPO data loader for Stage-2 training."""

import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from jsonschema import Draft202012Validator, ValidationError
from torch.utils.data import Dataset as TorchDataset

from NovelSFTDataset import NovelSFTDataset

DATA_SOURCES = ("real", "synthesized")


def _prompt_text(prompt: dict) -> str:
    parts = [prompt["user_request"]]
    if prompt["context"]:
        parts.append(f"上下文：{prompt['context']}")
    if prompt["original_text"]:
        parts.append(f"原文：{prompt['original_text']}")
    return "\n\n".join(parts)


def _normalize_sources(sources: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    sources = tuple(dict.fromkeys(sources))
    unknown = set(sources) - set(DATA_SOURCES)
    if not sources or unknown:
        raise ValueError(f"sources must contain one or more of {DATA_SOURCES}; got {sources}")
    return sources


def _record_paths(data_dir: str | Path, sources: tuple[str, ...] | list[str] = ("real",)) -> list[Path]:
    """Find records in explicitly selected source directories."""
    root = Path(data_dir)
    sources = _normalize_sources(sources)
    if root.name in DATA_SOURCES:
        if sources != (root.name,):
            raise ValueError(
                f"Direct data directory {root} requires sources=({root.name!r},); "
                "pass the parent data directory to select multiple sources"
            )
        record_dirs = [root]
    else:
        record_dirs = [root / source for source in sources]
    missing = [directory for directory in record_dirs if not directory.is_dir()]
    if missing:
        raise FileNotFoundError(f"Training data directory not found: {missing[0]}")
    paths = sorted(path for directory in record_dirs for path in directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No training records found in: {', '.join(map(str, record_dirs))}")
    return paths


def _local_records(data_dir: str | Path, sources: tuple[str, ...] | list[str] = ("real",)):
    """Validate schema and cross-field invariants; never guess or silently drop data."""
    schema_path = Path(__file__).resolve().parents[1] / "data" / "schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    record_ids, pair_ids = set(), set()
    for path in _record_paths(data_dir, sources):
        data_origin = "synthetic" if path.parent.name == "synthesized" else "real"
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
            validator.validate(bundle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error}") from error
        except ValidationError as error:
            raise ValueError(f"{path}: {error.json_path}: {error.message}") from error
        for item in bundle["records"]:
            location = f"{path}: record {item['id']!r}"
            declared_origin = item["metadata"].get("data_origin")
            if data_origin == "synthetic" and declared_origin != "synthetic":
                raise ValueError(f"{location}: synthesized records must declare data_origin='synthetic'")
            if data_origin == "real" and declared_origin not in (None, "real"):
                raise ValueError(f"{location}: real records cannot declare data_origin={declared_origin!r}")
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
            yield path, data_origin, item


def load_local_sft_records(data_dir: str | Path, sources: tuple[str, ...] | list[str] = ("real",)) -> list[dict]:
    """Project explicitly eligible canonical records into SFT samples."""
    return [
        {
            "id": item["id"],
            "source": path.name,
            "data_origin": data_origin,
            "prompt": _prompt_text(item["prompt"]),
            "response": item["sft"]["response"],
        }
        for path, data_origin, item in _local_records(data_dir, sources)
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


def load_local_dpo_records(data_dir: str | Path, sources: tuple[str, ...] | list[str] = ("real",)) -> list[dict]:
    """Pair the explicit chosen candidate with each eligible rejected candidate."""
    records = []
    for path, data_origin, item in _local_records(data_dir, sources):
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
                        "data_origin": data_origin,
                        "prompt": _prompt_text(item["prompt"]),
                        "chosen": chosen,
                        "rejected": candidate["response"],
                    }
                )
    return records


def sample_training_records(
    records: list[dict],
    source_weights: dict[str, float],
    sample_count: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample sources by relative weight, then sample uniformly within each source."""
    if not records:
        raise ValueError("records must not be empty")
    unknown = set(source_weights) - {"real", "synthetic"}
    if unknown or not source_weights or any(weight < 0 for weight in source_weights.values()):
        raise ValueError("source_weights must contain non-negative real/synthetic weights")
    grouped = {source: [record for record in records if record["data_origin"] == source] for source in source_weights}
    active = [source for source, weight in source_weights.items() if weight > 0]
    missing = [source for source in active if not grouped[source]]
    if not active or missing:
        raise ValueError(f"Positive-weight sources must contain records; missing: {missing}")
    if sample_count is None:
        sample_count = len(records)
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    rng = random.Random(seed)
    selected_sources = rng.choices(active, weights=[source_weights[source] for source in active], k=sample_count)
    return [rng.choice(grouped[source]) for source in selected_sources]


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
