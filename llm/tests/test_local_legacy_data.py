"""Regression tests for the real legacy editorial JSON layouts."""
import json
from pathlib import Path

from LocalNovelDataset import load_local_dpo_records, load_local_sft_records


def _write(directory, name, payload):
    (directory / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_compact_legacy_focus_is_prompt(tmp_path):
    _write(tmp_path, "legacy.json", {
        "scene": "第三章出院对白",
        "sft": [{"instruction": "简化对白", "chosen": "正确对白"}],
        "dpo": [{"focus": "先说明换药安排", "chosen": "先告知换药", "rejected": "先问换药"}],
    })
    sft = load_local_sft_records(tmp_path)
    dpo = load_local_dpo_records(tmp_path)
    assert len(sft) == 1
    assert len(dpo) == 1
    assert "先说明换药安排" in dpo[0]["prompt"]
    assert dpo[0]["chosen"] == "先告知换药"


def test_legacy_rejected_array_expands_all_verbatim_alternatives(tmp_path):
    _write(tmp_path, "ward.json", {
        "description": "病房对白",
        "records": [{"id": "medical_status", "sft": {"eligible": True, "response": "作者采用的完整对白"},
                     "preference": {"eligible": True, "rejected": ["机械短句", "多余观察动作"]}}],
    })
    dpo = load_local_dpo_records(tmp_path)
    assert len(dpo) == 2
    assert {x["rejected"] for x in dpo} == {"机械短句", "多余观察动作"}
    assert all(x["chosen"] == "作者采用的完整对白" for x in dpo)


def test_explicit_composite_candidate_and_multiple_rejections(tmp_path):
    _write(tmp_path, "composite.json", {
        "records": [{"id": "combined", "prompt": {"user_request": "按作者选择组合"},
                     "preference": {"eligible": True, "chosen_candidate_id": "C+A",
                         "candidates": [
                             {"candidate_id": "A", "status": "partially_chosen", "response": "原A"},
                             {"candidate_id": "B", "status": "rejected", "response": "原B"},
                             {"candidate_id": "C", "status": "partially_chosen", "response": "原C"},
                             {"candidate_id": "C+A", "status": "chosen", "response": "原C与原A的明确组合"}]}}],
    })
    dpo = load_local_dpo_records(tmp_path)
    assert len(dpo) == 1
    assert dpo[0]["chosen"] == "原C与原A的明确组合"


def test_empty_or_identical_pair_not_trainable(tmp_path):
    _write(tmp_path, "invalid.json", {
        "scene": "场景",
        "dpo": [{"chosen": "同文", "rejected": "同文"},
                {"chosen": "有正文", "rejected": ""}],
    })
    assert load_local_dpo_records(tmp_path) == []
