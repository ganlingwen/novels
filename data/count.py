#!/usr/bin/env python3
"""Count exactly the records emitted by llm/LocalNovelDataset.py.

Run from any working directory: python data/count.py
"""

from collections import Counter
import json
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

    # Keep the historical annotation count visible; do not silently replace it
    # with the loader result. A gap is a data-quality or compatibility issue.
    annotated_pairs = 0
    for path in files:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        annotated_pairs += len(bundle.get("dpo", []))
        for item in bundle.get("records", [bundle]):
            preference = item.get("preference", {})
            if preference.get("eligible") is not True:
                continue
            if isinstance(preference.get("rejected"), list) and item.get("sft", {}).get("eligible") is True:
                annotated_pairs += len(preference["rejected"])
                continue
            chosen_id = preference.get("chosen_candidate_id")
            candidates = preference.get("candidates", [])
            if chosen_id and any(c.get("candidate_id") == chosen_id for c in candidates):
                annotated_pairs += sum(c.get("status") == "rejected"
                                       and c.get("candidate_id") != chosen_id for c in candidates)
    print(f"JSON files: {len(files)}")
    print(f"DPO pairs (annotated, historical count.py rules): {annotated_pairs}")
    print(f"DPO pairs (loader): {len(dpo_records)}")
    print(f"Difference (annotated - loader): {annotated_pairs - len(dpo_records)}")
    print(f"SFT samples (loaded): {len(sft_records)}")
    print("Per-file loaded counts (SFT / DPO):")
    for path in files:
        print(f"  {path.name}: {sft_by_file[path.name]} / {dpo_by_file[path.name]}")


if __name__ == "__main__":
    main()
