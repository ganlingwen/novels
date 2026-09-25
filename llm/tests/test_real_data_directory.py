"""Regression tests for the real-record directory migration."""

import json

import pytest

from LocalNovelDataset import _record_paths, load_local_dpo_records, load_local_sft_records


def test_real_directory_is_discovered_from_parent_and_direct_path(tmp_path):
    real = tmp_path / "data" / "real"
    real.mkdir(parents=True)
    (tmp_path / "data" / "schema.json").write_text("{}", encoding="utf-8")
    (real / "sample.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "metadata": {},
                "records": [
                    {
                        "id": "sample",
                        "prompt": {
                            "user_request": "修改这段文字",
                            "context": None,
                            "original_text": None,
                            "metadata": {},
                        },
                        "sft": {"eligible": True, "response": "修改后的文字", "metadata": {}},
                        "dpo": {
                            "eligible": True,
                            "candidates": [
                                {
                                    "candidate_id": "A",
                                    "response": "修改后的文字",
                                    "status": "chosen",
                                    "pair_id": None,
                                    "metadata": {},
                                },
                                {
                                    "candidate_id": "B",
                                    "response": "另一种写法",
                                    "status": "rejected",
                                    "pair_id": "sample:B",
                                    "metadata": {},
                                },
                            ],
                            "chosen_candidate_id": "A",
                            "metadata": {},
                        },
                        "metadata": {},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for directory in (tmp_path / "data", real):
        assert len(load_local_sft_records(directory)) == 1
        assert len(load_local_dpo_records(directory)) == 1
    assert [p.name for p in _record_paths(tmp_path / "data")] == ["sample.json"]


def test_missing_real_directory_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local_sft_records(tmp_path / "data")
    with pytest.raises(FileNotFoundError):
        load_local_dpo_records(tmp_path / "data")


def test_empty_real_directory_fails(tmp_path):
    real = tmp_path / "data" / "real"
    real.mkdir(parents=True)
    with pytest.raises(ValueError, match="No training records"):
        load_local_sft_records(tmp_path / "data")
    with pytest.raises(ValueError, match="No training records"):
        load_local_dpo_records(tmp_path / "data")
