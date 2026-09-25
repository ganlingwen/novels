"""Canonical contract, real-corpus migration, and invalid-data regressions."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from LocalNovelDataset import load_local_dpo_records, load_local_sft_records, sample_training_records

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


def _write(tmp_path, records, name="sample.json", source="real"):
    record_dir = tmp_path / source
    record_dir.mkdir(exist_ok=True)
    (record_dir / name).write_text(
        json.dumps({"schema_version": "3.0", "metadata": {}, "records": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return record_dir


def test_explicit_selection_keeps_sft_and_dpo_targets_independent(tmp_path, record):
    real = _write(tmp_path, [record])
    sft = load_local_sft_records(real)
    dpo = load_local_dpo_records(real)
    assert sft == [
        {
            "id": "edit-1",
            "source": "sample.json",
            "data_origin": "real",
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


def _without_origin(records):
    return [{key: value for key, value in record.items() if key != "data_origin"} for record in records]


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
    assert _digest(_without_origin([r for r in sft if r["id"] not in RECOVERED_IDS])) == (
        "49dcb05972816fd968f994d606c78f7da38c7006ec860d903d67ed353d315426"
    )
    assert _digest(_without_origin(dpo)) == "1fe954d442f5bf5387c8f305cf2634c5318f479937cbf7251948a8b867fa2c29"


def test_reviewer_only_record_does_not_become_novel_prose_sft():
    name = "rejected_pr1_tiangan-ability-false-positive.json"
    assert not any(r["source"] == name for r in load_local_sft_records(DATA_DIR))
    assert len([r for r in load_local_dpo_records(DATA_DIR) if r["source"] == name]) == 1


def test_source_weight_sampling_is_deterministic_and_source_level():
    records = [
        {"id": "real-1", "data_origin": "real"},
        {"id": "real-2", "data_origin": "real"},
        {"id": "synthetic-1", "data_origin": "synthetic"},
    ]
    first = sample_training_records(records, {"real": 1, "synthetic": 3}, sample_count=1000, seed=7)
    second = sample_training_records(records, {"real": 1, "synthetic": 3}, sample_count=1000, seed=7)
    assert first == second
    synthetic_count = sum(record["data_origin"] == "synthetic" for record in first)
    assert 700 < synthetic_count < 800


def test_synthesized_record_requires_full_provenance_and_explicit_source(tmp_path, record):
    real_record = deepcopy(record)
    real_record["id"] = "real-edit-1"
    real_record["dpo"]["candidates"][1]["pair_id"] = "real-pair-1"
    real_record["dpo"]["candidates"][3]["pair_id"] = "real-pair-2"
    _write(tmp_path, [real_record])

    record["id"] = "synthetic-edit-1"
    record["dpo"]["candidates"][1]["pair_id"] = "synthetic-pair-1"
    record["dpo"]["candidates"][3]["pair_id"] = "synthetic-pair-2"
    record["metadata"] = {
        "data_origin": "synthetic",
        "synthetic_provenance": {
            "parent_record_ids": ["real-parent-1"],
            "source_group_id": "chapter3-ward",
            "synthesis_method": "preference_transfer",
            "fact_sources": ["characters.md#白川", "generated/chapter3_scene.md"],
            "generator": {
                "model": "generator-model",
                "prompt_template_version": "v1",
                "sampling": {"temperature": 0.8},
            },
            "judge": {
                "model": "judge-model",
                "rubric_version": "v1",
                "decision": "accepted",
            },
            "audit_status": "machine_reviewed",
        },
    }
    synthesized = _write(tmp_path, [record], source="synthesized")
    loaded = load_local_sft_records(tmp_path, sources=("synthesized",))
    assert loaded[0]["data_origin"] == "synthetic"
    assert load_local_sft_records(synthesized, sources=("synthesized",)) == loaded
    both = load_local_sft_records(tmp_path, sources=("real", "synthesized"))
    assert [item["data_origin"] for item in both] == ["real", "synthetic"]

    del record["metadata"]["synthetic_provenance"]
    _write(tmp_path, [record], source="synthesized")
    with pytest.raises(ValueError, match="synthetic_provenance"):
        load_local_sft_records(tmp_path, sources=("synthesized",))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"synthetic": 1}, "missing"),
        ({"real": -1}, "non-negative"),
        ({}, "non-negative"),
    ],
)
def test_source_weight_sampling_rejects_invalid_configuration(weights, message):
    with pytest.raises(ValueError, match=message):
        sample_training_records([{"id": "real-1", "data_origin": "real"}], weights)


@pytest.mark.parametrize("source", ["real", "synthesized"])
@pytest.mark.parametrize("loader", [load_local_sft_records, load_local_dpo_records])
def test_direct_directory_cannot_silently_drop_selected_source(tmp_path, source, loader):
    directory = tmp_path / source
    directory.mkdir()
    with pytest.raises(ValueError, match="parent data directory"):
        loader(directory, sources=("real", "synthesized"))


@pytest.mark.parametrize("decision", ["accepted", "rejected", "tie"])
@pytest.mark.parametrize("sft_eligible,dpo_eligible", [(True, True), (True, False), (False, True), (False, False)])
def test_synthetic_judgment_controls_training_eligibility(tmp_path, record, decision, sft_eligible, dpo_eligible):
    record["metadata"] = {
        "data_origin": "synthetic",
        "synthetic_provenance": {
            "parent_record_ids": ["parent"],
            "source_group_id": "group",
            "synthesis_method": "new_edit",
            "fact_sources": ["characters.md"],
            "generator": {"model": "generator", "prompt_template_version": "v1", "sampling": {}},
            "judge": {"model": "judge", "rubric_version": "v1", "decision": decision},
            "audit_status": "machine_reviewed",
        },
    }
    record["sft"]["eligible"] = sft_eligible
    record["dpo"]["eligible"] = dpo_eligible
    if not dpo_eligible:
        for candidate in record["dpo"]["candidates"]:
            candidate["pair_id"] = None
    _write(tmp_path, [record], source="synthesized")
    bundle = {"schema_version": "3.0", "metadata": {}, "records": [record]}
    validator = Draft202012Validator(json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8")))
    invalid = decision != "accepted" and (sft_eligible or dpo_eligible)
    assert validator.is_valid(bundle) is not invalid
    for loader, count in ((load_local_sft_records, int(sft_eligible)), (load_local_dpo_records, 2 * int(dpo_eligible))):
        if invalid:
            with pytest.raises(ValueError, match="accepted"):
                loader(tmp_path, sources=("synthesized",))
        else:
            assert len(loader(tmp_path, sources=("synthesized",))) == count
