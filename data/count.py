#!/usr/bin/env python3
"""Count exactly the records emitted by llm/LocalNovelDataset.py.

Run from any working directory: python data/count.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

EDITORIAL_TAGS = {
    "continuity",
    "plot",
    "spatial_logic",
    "dialogue",
    "cinematic",
    "tension",
    "presentation",
    "pov",
    "foreshadowing",
    "everyday_life",
}


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(data_dir.parent / "llm"))
    from LocalNovelDataset import load_local_dpo_records, load_local_sft_records

    files = sorted((data_dir / "real").glob("*.json"))
    sft_records = load_local_sft_records(data_dir)
    dpo_records = load_local_dpo_records(data_dir)
    sft_by_file = Counter(record["source"] for record in sft_records)
    dpo_by_file = Counter(record["source"] for record in dpo_records)

    # Count annotations independently of the loader projection.
    annotated_pairs = 0
    primary_tags = Counter()
    secondary_tags = Counter()
    annotation_files = files + sorted((data_dir / "review").glob("*.json"))
    for path in files:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for item in bundle["records"]:
            dpo = item["dpo"]
            if dpo["eligible"]:
                annotated_pairs += sum(
                    c["status"] == "rejected" for c in dpo["candidates"]
                )
    for path in annotation_files:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for item in bundle["records"]:
            tags = item["metadata"]["tags"]
            primary = tags["primary"]
            secondary = tags["secondary"]
            if primary not in EDITORIAL_TAGS or any(tag not in EDITORIAL_TAGS for tag in secondary):
                raise ValueError(f"{path.name}: unknown editorial tag in {item['id']}")
            if primary in secondary or len(secondary) != len(set(secondary)):
                raise ValueError(f"{path.name}: duplicate editorial tag in {item['id']}")
            primary_tags[primary] += 1
            secondary_tags.update(secondary)
    print(f"JSON files: {len(files)}")
    print(f"DPO pairs (annotated): {annotated_pairs}")
    print(f"DPO pairs (loader): {len(dpo_records)}")
    print(f"Difference (annotated - loader): {annotated_pairs - len(dpo_records)}")
    print(f"SFT samples (loaded): {len(sft_records)}")
    print(f"Tagged records (real + review): {sum(primary_tags.values())}")
    print("Tag distribution (primary / secondary):")
    for tag in sorted(EDITORIAL_TAGS):
        print(f"  {tag}: {primary_tags[tag]} / {secondary_tags[tag]}")
    print("Per-file loaded counts (SFT / DPO):")
    for path in files:
        print(f"  {path.name}: {sft_by_file[path.name]} / {dpo_by_file[path.name]}")


if __name__ == "__main__":
    main()
