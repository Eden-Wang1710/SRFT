#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


DEFAULT_INPUT_FILES = [
    "toucan_hotel_32B_v1.json",
    "toucan_hotel_32B_v1.5.json",
    "toucan_hotel_32B_v2.json",
    "toucan_hotel_v3.json",
    
    "toucan_email_32B_v1.json",
    "toucan_email_32B_v1.5.json",
    "toucan_email_32B_v2.json",
    "toucan_email_v3.json",
    
    "toucan_windows_32B_v1.json",
    "toucan_windows_32B_v1.5.json",
    "toucan_windows_32B_v2.json",
    
    "toucan_downloader_32B_v1.json",
    "toucan_downloader_32B_v1.5.json",
    "toucan_downloader_32B_v2.json",
    
    "toucan_Minecraft_32B_v1.json",
    "toucan_Minecraft_32B_v1.5.json",
    "toucan_Minecraft_32B_v2.json",
    
    # v3.2
    "toucan_ennkaheksa_v3.json",
    "toucan_word_v3.json",
    
    
]

DATASET_INFO_TEMPLATE = {
    "file_name": "toucan_32B_v3.2.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system"},
    "tags": {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "system_tag": "system",
        "function_tag": "function_call",
        "observation_tag": "observation",
    },
}


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and shuffle Toucan v1 datasets into toucan_32B_v2."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing source datasets and output file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="toucan_32B_v3.2.json",
        help="Output file name under data-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling.",
    )
    args = parser.parse_args()

    merged: list[dict] = []
    counts: dict[str, int] = {}

    for file_name in DEFAULT_INPUT_FILES:
        file_path = args.data_dir / file_name
        records = load_records(file_path)
        merged.extend(records)
        counts[file_name] = len(records)

    random.Random(args.seed).shuffle(merged)

    output_path = args.data_dir / args.output_file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print("Merged datasets:")
    for file_name in DEFAULT_INPUT_FILES:
        print(f"- {file_name}: {counts[file_name]}")
    print(f"Total: {len(merged)}")
    print(f"Output: {output_path}")
    print(f"Seed: {args.seed}")
    print("\nPaste this into data/dataset_info.json:")
    print(json.dumps({"toucan_32B_v2": DATASET_INFO_TEMPLATE}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
