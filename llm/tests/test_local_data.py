"""Canonical contract, real-corpus migration, and invalid-data regressions."""

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from LocalNovelDataset import load_local_dpo_records, load_local_sft_records

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RECOVERED_IDS = {
    "ch3_baichuan_medical_bill_20260919:final_context_confirmation",
    "ch3_cat_kill_credit_20260919:final_context_confirmation",
}


def _candidate(cid, response, status, pair_id=None):
    return {
        "candidate_id": cid,
        "response": response,
        "status": status,
        "pair_id": pair_id,
        "metadata": {},
    }


@pytest.fixture
def record():
    return {
        "id": "edit-1",
        "prompt": {
            "user_request": "修改对白",
            "context": "病房",
            "original_text": "原文",
            "metadata": {"source": "explicit_author_request"},
        },
        "sft": {"eligible": True, "response": "作者最终修订", "metadata": {}},
        "dpo": {
            "eligible": True,
            "chosen_candidate_id": "C+A",
            "candidates": [
                _candidate("A", "第一轮候选", "chosen_then_refined"),
                _candidate("B", "另一完整候选", "rejected", "pair-1"),
                _candidate("C+A", "作者选择的组合", "chosen"),
                _candidate("D", "另一轮被拒稿", "rejected", "pair-2"),
                _candidate("component", "仅被拒绝作为完整版本", "rejected_as_full_candidate"),
            ],
            "metadata": {"selection_source": "explicit_author_choice"},
        },
        "metadata": {"review": {"edit_instruction": "不应被隐式加入 prompt"}},
    }


def _write(tmp_path, records, name="sample.json"):
    real = tmp_path / "real"
    real.mkdir(exist_ok=True)
    (real / name).write_text(
        json.dumps({"schema_version": "3.0", "metadata": {}, "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return real


def test_explicit_selection_keeps_sft_and_dpo_targets_independent(tmp_path, record):
    real = _write(tmp_path, [record])
    sft = load_local_sft_records(real)
    dpo = load_local_dpo_records(real)
    assert sft == [
        {
            "id": "edit-1",
            "source": "sample.json",
            "prompt": "修改对白\n\n上下文：病房\n\n原文：原文",
            "response": "作者最终修订",
        }
    ]
    assert [r["id"] for r in dpo] == ["pair-1", "pair-2"]
    assert [r["rejected"] for r in dpo] == ["另一完整候选", "另一轮被拒稿"]
    assert all(r["chosen"] == "作者选择的组合" and r["prompt"] == sft[0]["prompt"] for r in dpo)


def test_ineligible_annotations_are_preserved_but_not_trained(tmp_path, record):
    record["sft"]["eligible"] = False
    record["dpo"]["eligible"] = False
    for candidate in record["dpo"]["candidates"]:
        candidate["pair_id"] = None
    real = _write(tmp_path, [record])
    assert load_local_sft_records(real) == []
    assert load_local_dpo_records(real) == []


@pytest.mark.parametrize(
    "failure",
    [
        "missing_prompt",
        "blank_response",
        "unknown_chosen",
        "duplicate_candidate",
        "identical_pair",
        "missing_pair_id",
        "duplicate_pair_id",
        "untrained_pair_id",
        "legacy_response_alias",
        "legacy_rejected_array",
    ],
)
def test_malformed_records_fail_with_file_and_record_location(tmp_path, record, failure):
    dpo = record["dpo"]
    if failure == "missing_prompt":
        del record["prompt"]
    elif failure == "blank_response":
        record["sft"]["response"] = " \n"
    elif failure == "unknown_chosen":
        dpo["chosen_candidate_id"] = "unknown"
    elif failure == "duplicate_candidate":
        dpo["candidates"].append(dpo["candidates"][0].copy())
    elif failure == "identical_pair":
        dpo["candidates"][1]["response"] = "作者选择的组合"
    elif failure == "missing_pair_id":
        dpo["candidates"][1]["pair_id"] = None
    elif failure == "duplicate_pair_id":
        dpo["candidates"][3]["pair_id"] = "pair-1"
    elif failure == "untrained_pair_id":
        dpo["candidates"][0]["pair_id"] = "unexpected"
    elif failure == "legacy_response_alias":
        record["sft"]["output"] = record["sft"].pop("response")
    elif failure == "legacy_rejected_array":
        dpo["rejected"] = ["旧格式"]
    real = _write(tmp_path, [record])
    for loader in (load_local_sft_records, load_local_dpo_records):
        with pytest.raises(ValueError, match=r"sample.json: .*record"):
            loader(real)


@pytest.mark.parametrize("duplicate", ["record", "pair"])
def test_duplicate_ids_across_files_fail_instead_of_dropping_samples(tmp_path, record, duplicate):
    real = _write(tmp_path, [record], "a.json")
    if duplicate == "pair":
        record["id"] = "edit-2"
    _write(tmp_path, [record], "b.json")
    with pytest.raises(ValueError, match=f"duplicate {duplicate} ID"):
        load_local_sft_records(real)


def test_old_layout_is_rejected(tmp_path):
    real = _write(tmp_path, [])
    (real / "sample.json").write_text(json.dumps({"sft": [{"instruction": "旧", "output": "稿"}]}))
    with pytest.raises(ValueError, match="sample.json"):
        load_local_sft_records(real)


def _digest(records):
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def test_entire_real_corpus_matches_schema_and_preserves_migration_baseline():
    schema = json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    paths = sorted((DATA_DIR / "real").glob("*.json"))
    assert len(paths) == 76
    for path in paths:
        validator.validate(json.loads(path.read_text(encoding="utf-8")))

    sft = load_local_sft_records(DATA_DIR)
    dpo = load_local_dpo_records(DATA_DIR)
    assert len(sft) == 176
    assert len(dpo) == 230
    assert {r["id"] for r in sft} & RECOVERED_IDS == RECOVERED_IDS
    # Captured from the previous loader before migrating: protects text, literal
    # escapes, prompts, order, source filenames, and all existing training IDs.
    assert _digest([r for r in sft if r["id"] not in RECOVERED_IDS]) == (
        "49dcb05972816fd968f994d606c78f7da38c7006ec860d903d67ed353d315426"
    )
    assert _digest(dpo) == "1fe954d442f5bf5387c8f305cf2634c5318f479937cbf7251948a8b867fa2c29"


def test_reviewer_only_record_does_not_become_novel_prose_sft():
    name = "rejected_pr1_tiangan-ability-false-positive.json"
    assert not any(r["source"] == name for r in load_local_sft_records(DATA_DIR))
    assert len([r for r in load_local_dpo_records(DATA_DIR) if r["source"] == name]) == 1
