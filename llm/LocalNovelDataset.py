"""Local manual SFT data loader for Stage-2 training."""

import hashlib
import json
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from NovelSFTDataset import NovelSFTDataset


def _prompt_text(prompt: dict) -> str:
    parts = [prompt.get("user_request") or ""]
    if prompt.get("context"):
        parts.append(f"上下文：{prompt['context']}")
    original = prompt.get("original_text") or prompt.get("original")
    if original:
        parts.append(f"原文：{original}")
    return "\n\n".join(parts)


def _record_id(path: Path, index: int, answer: str) -> str:
    digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:12]
    return f"{path.name}:{index}:{digest}"


def load_local_sft_records(data_dir: str | Path) -> list[dict]:
    """Normalize schema-2 and compact bundles into local SFT records."""
    records = []
    seen = set()
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "schema.json":
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(bundle.get("sft"), list):
            for index, item in enumerate(bundle["sft"]):
                prompt = item.get("instruction")
                answer = item.get("output") or item.get("response") or item.get("chosen")
                if not prompt or not answer:
                    continue
                rid = _record_id(path, index, answer)
                if rid not in seen:
                    seen.add(rid)
                    records.append({"id": rid, "source": path.name, "prompt": prompt, "response": answer})
            continue

        for index, item in enumerate(bundle.get("records", [bundle])):
            sft = item.get("sft", {})
            prompt = item.get("prompt", {})
            answer = sft.get("response")
            request = _prompt_text(prompt)
            if sft.get("eligible") is not True or not request or not answer:
                continue
            rid = item.get("id") or _record_id(path, index, answer)
            if rid not in seen:
                seen.add(rid)
                records.append({"id": rid, "source": path.name, "prompt": request, "response": answer})
    return records


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


def _preference_candidates(item: dict) -> tuple[str | None, list[str]]:
    preference = item.get("preference", {})
    if preference.get("eligible") is not True:
        return None, []
    candidates = preference.get("candidates", [])
    chosen_id = preference.get("chosen_candidate_id")
    chosen = next(
        (
            candidate.get("response")
            for candidate in candidates
            if candidate.get("candidate_id") == chosen_id
            or candidate.get("status") in ("chosen", "chosen_then_refined")
        ),
        None,
    )
    rejected = [candidate.get("response") for candidate in candidates if candidate.get("status") == "rejected"]
    return chosen, [response for response in rejected if response]


def load_local_dpo_records(data_dir: str | Path) -> list[dict]:
    """Load real same-prompt preference pairs from compact and schema-2 records."""
    records = []
    seen = set()
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "schema.json":
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))

        for index, item in enumerate(bundle.get("dpo", [])):
            prompt = item.get("prompt")
            chosen = item.get("chosen")
            rejected = item.get("rejected")
            if not prompt or not chosen or not rejected:
                continue
            rid = f"{path.name}:dpo:{index}:{hashlib.sha256((chosen + '\\n' + rejected).encode()).hexdigest()[:12]}"
            if rid not in seen:
                seen.add(rid)
                records.append(
                    {"id": rid, "source": path.name, "prompt": prompt, "chosen": chosen, "rejected": rejected}
                )

        for index, item in enumerate(bundle.get("records", [bundle])):
            chosen, rejected = _preference_candidates(item)
            prompt = _prompt_text(item.get("prompt", {}))
            if not prompt or not chosen or not rejected:
                continue
            base_id = item.get("id") or f"{path.name}:record:{index}"
            for rejected_index, rejected_response in enumerate(rejected):
                rid = f"{base_id}:rejected:{rejected_index}"
                if rid not in seen:
                    seen.add(rid)
                    records.append(
                        {
                            "id": rid,
                            "source": path.name,
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected_response,
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
