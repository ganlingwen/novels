"""Local manual SFT data loader for Stage-2 training."""

import hashlib
import json
from pathlib import Path

from datasets import Dataset

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
