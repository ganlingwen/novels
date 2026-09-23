import sys

from SFTTrainer import create_run_directory, tensorboard_directory
from train_qwen3_4b_sft import parse_args


def test_max_steps_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_qwen3_4b_sft.py", "--max-steps", "9000"])

    args = parse_args()

    assert args.max_steps == 9000
    assert not hasattr(args, "epochs")


def test_new_runs_get_unique_directories(tmp_path):
    first = create_run_directory(str(tmp_path))
    second = create_run_directory(str(tmp_path))

    assert first.parent == tmp_path / "runs"
    assert second.parent == tmp_path / "runs"
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_resume_uses_checkpoint_run_directory(tmp_path):
    checkpoint = tmp_path / "runs" / "run-1" / "checkpoints" / "step-001000"
    checkpoint.mkdir(parents=True)

    assert create_run_directory(str(tmp_path), str(checkpoint)) == checkpoint.parent.parent


def test_tensorboard_uses_run_name(tmp_path):
    run = tmp_path / "runs" / "20260918-180919"

    assert tensorboard_directory(str(run)) == tmp_path / "tensorboard" / "20260918-180919"
