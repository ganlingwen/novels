#!/usr/bin/env python3
"""Unified compatibility entry point for the PyTorch training objectives."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--objective", choices=("sft", "stage2-sft", "dpo"), default="sft")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.objective == "sft":
        from train_qwen3_4b_sft import main as train_main
    elif args.objective == "stage2-sft":
        from train_stage2_sft import main as train_main
    else:
        from DPOTrainer import main as train_main
    train_main()


if __name__ == "__main__":
    main()
