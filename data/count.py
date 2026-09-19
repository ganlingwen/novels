#!/usr/bin/env python3
"""Count eligible SFT samples and DPO pairs in this directory.

Run from any working directory: python data/count.py
"""

import json
from pathlib import Path


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    files = sorted(path for path in data_dir.glob("*.json") if path.name != "schema.json")
    sft = dpo_records = dpo_pairs = 0

    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("sft", {}).get("eligible") is True:
            sft += 1

        preference = obj.get("preference", {})
        if preference.get("eligible") is not True:
            continue

        chosen_id = preference.get("chosen_candidate_id")
        candidates = preference.get("candidates", [])
        if not chosen_id or not any(
            candidate.get("candidate_id") == chosen_id
            for candidate in candidates
        ):
            continue

        rejected_count = sum(
            candidate.get("status") == "rejected"
            and candidate.get("candidate_id") != chosen_id
            for candidate in candidates
        )
        if rejected_count:
            dpo_records += 1
            dpo_pairs += rejected_count

    print(f"JSON files: {len(files)}")
    print(f"SFT samples: {sft}")
    print(f"DPO records: {dpo_records}")
    print(f"DPO pairs: {dpo_pairs}")


if __name__ == "__main__":
    main()
