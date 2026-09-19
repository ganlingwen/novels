import sys

from train_qwen3_4b_sft import parse_args


def test_max_steps_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_qwen3_4b_sft.py", "--max-steps", "9000"])

    args = parse_args()

    assert args.max_steps == 9000
    assert not hasattr(args, "epochs")
