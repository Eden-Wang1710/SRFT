import argparse
import json
from pathlib import Path


def get_injection_task_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("injection_task_"):
            return part
    return "unknown_injection_task"


def iter_records(path: Path):
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            yield obj
        return

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
                if isinstance(obj, dict):
                    yield obj


def get_record_bools(obj: dict, field_name: str) -> list[bool]:
    """Collect bool labels from both legacy top-level and per-sample formats."""
    values: list[bool] = []

    top_value = obj.get(field_name)
    if isinstance(top_value, bool):
        values.append(top_value)

    qwen_samples = obj.get("qwen_samples")
    if isinstance(qwen_samples, list):
        for sample in qwen_samples:
            if not isinstance(sample, dict):
                continue
            sample_value = sample.get(field_name)
            if isinstance(sample_value, bool):
                values.append(sample_value)

    return values


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count how many records have follows_injection_task_action=True "
            "under a directory containing .json/.jsonl files."
        )
    )
    parser.add_argument("root", type=str, help="Root directory to scan.")
    parser.add_argument(
        "--field-name",
        type=str,
        default="follows_injection_task_action",
        help="Boolean field name to count.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*",
        help="Path glob under root (default: **/*).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    files = [p for p in root.glob(args.glob) if p.is_file() and p.suffix in {".json", ".jsonl"}]
    files.sort()

    total_records = 0
    true_records = 0
    valued_records = 0
    files_with_records = 0
    files_with_true = 0
    files_with_valued = 0
    per_task_stats = {}

    for path in files:
        file_records = 0
        file_has_true = False
        file_has_valued = False
        task_name = get_injection_task_name(path)
        task_stats = per_task_stats.setdefault(
            task_name,
            {"total": 0, "valued": 0, "true": 0},
        )
        for obj in iter_records(path):
            file_records += 1
            total_records += 1
            task_stats["total"] += 1
            bool_values = get_record_bools(obj, args.field_name)
            if bool_values:
                valued_records += 1
                task_stats["valued"] += 1
                file_has_valued = True
            if any(bool_values):
                true_records += 1
                task_stats["true"] += 1
                file_has_true = True
        if file_records > 0:
            files_with_records += 1
        if file_has_valued:
            files_with_valued += 1
        if file_has_true:
            files_with_true += 1

    ratio = (true_records / valued_records) if valued_records else 0.0
    print(f"Root: {root}")
    print(f"Files scanned (.json/.jsonl): {len(files)}")
    print(f"Files with records: {files_with_records}")
    print(f"Files with {args.field_name} (legacy or per-sample bool): {files_with_valued}")
    print(f"Files with any sample {args.field_name}=True: {files_with_true}")
    print(f"Total records: {total_records}")
    print(f"{args.field_name} has value (bool): {valued_records}")
    print(f"{args.field_name} missing/non-bool: {total_records - valued_records}")
    print(f"{args.field_name}=True: {true_records}")
    print(f"Ratio (True / valued): {ratio:.4f}")

    print("\nDistribution by injection task:")
    def sort_key(task_name: str):
        prefix = "injection_task_"
        if task_name.startswith(prefix):
            suffix = task_name[len(prefix) :]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, task_name)

    for task_name in sorted(per_task_stats, key=sort_key):
        stats = per_task_stats[task_name]
        task_total = stats["total"]
        task_valued = stats["valued"]
        task_true = stats["true"]
        task_false = task_valued - task_true
        task_missing = task_total - task_valued
        task_ratio = (task_true / task_valued) if task_valued else 0.0
        print(
            f"- {task_name}: total={task_total}, true={task_true}, false={task_false}, "
            f"missing/non-bool={task_missing}, ratio_true={task_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
