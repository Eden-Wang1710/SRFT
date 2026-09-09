import argparse
import json
from pathlib import Path


def compute_injection_inserted(data: dict) -> bool:
    context_messages = data.get("context_messages") or []
    context_len = len(context_messages)
    injections = data.get("injections") or {}

    targets = []
    for inj in injections.values():
        if not isinstance(inj, dict):
            continue

        # New format: one injection can have multiple insertions.
        insertions = inj.get("insertions")
        if isinstance(insertions, list):
            for ins in insertions:
                if not isinstance(ins, dict):
                    continue
                idx = ins.get("target_message_index")
                if isinstance(idx, int):
                    targets.append(idx)
            continue

        # Backward compatibility with the old single-target format.
        idx = inj.get("target_message_index")
        if isinstance(idx, int):
            targets.append(idx)

    if not targets:
        return False

    return any(idx < context_len for idx in targets)


def iter_step_files(root: Path):
    return sorted(root.rglob("assistant_step_*.json"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Traverse assistant_step JSON files and add a boolean field indicating "
            "whether injection has been inserted by this step."
        )
    )
    parser.add_argument("root", type=str, help="Root path, e.g. agentdojo/runs/qwen3-8b-think-samples-toucan_hotel_v1")
    parser.add_argument(
        "--field-name",
        type=str,
        default="has_injection_inserted_by_step",
        help="Field name to write into each JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned changes; do not write files.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only write when the target field does not already exist.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    files = iter_step_files(root)
    if not files:
        print(f"No assistant_step JSON files found under: {root}")
        return

    written = 0
    skipped_existing = 0
    true_count = 0

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        value = compute_injection_inserted(data)
        if value:
            true_count += 1

        if args.only_missing and args.field_name in data:
            skipped_existing += 1
            continue

        old_value = data.get(args.field_name, None)
        if old_value is value:
            continue

        if args.dry_run:
            print(f"[dry-run] {path}: {args.field_name} {old_value!r} -> {value}")
            written += 1
            continue

        data[args.field_name] = value
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        written += 1

    print(f"Scanned: {len(files)}")
    print(f"Injection-inserted=True: {true_count}")
    print(f"{'Would write' if args.dry_run else 'Written'}: {written}")
    if args.only_missing:
        print(f"Skipped existing field: {skipped_existing}")


if __name__ == "__main__":
    main()
