#!/usr/bin/env python3
"""Count exactly the records emitted by llm/LocalNovelDataset.py.

Run from any working directory: python data/count.py
"""

from collections import Counter
from pathlib import Path
import sys


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(data_dir.parent / "llm"))
    from LocalNovelDataset import load_local_dpo_records, load_local_sft_records

    files = sorted(path for path in data_dir.glob("*.json") if path.name != "schema.json")
    sft_records = load_local_sft_records(data_dir)
    dpo_records = load_local_dpo_records(data_dir)
    sft_by_file = Counter(record["source"] for record in sft_records)
    dpo_by_file = Counter(record["source"] for record in dpo_records)

    print(f"JSON files: {len(files)}")
    print(f"SFT samples (loaded): {len(sft_records)}")
    print(f"DPO pairs (loaded): {len(dpo_records)}")
    print("Per-file loaded counts (SFT / DPO):")
    for path in files:
        print(f"  {path.name}: {sft_by_file[path.name]} / {dpo_by_file[path.name]}")


if __name__ == "__main__":
    main()
