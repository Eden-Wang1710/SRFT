#!/usr/bin/env python3
import argparse
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


DEFAULT_ROOT = "CoT/runs/qwen3-8b-think-samples-toucan_Minecraft_v1"
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average token count of .txt files using a Qwen tokenizer."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_ROOT,
        help=f"Root directory to scan recursively (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer name or local path (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load tokenizer from local files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    txt_files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if not txt_files:
        print(f"No .txt files found under: {root}")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    token_counts = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(ids))

    total_files = len(token_counts)
    total_tokens = sum(token_counts)
    avg_tokens = mean(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    print(f"Root: {root}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Files: {total_files}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per .txt: {avg_tokens:.4f}")
    print(f"Min tokens: {min_tokens}")
    print(f"Max tokens: {max_tokens}")


if __name__ == "__main__":
    main()
