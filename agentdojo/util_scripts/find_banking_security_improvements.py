#!/usr/bin/env python3
"""Find banking JSONs where baseline security=False but safe-agent security=True."""

from pathlib import Path
import json
from typing import Optional


BASELINE_ROOT = Path("runs/Qwen_Qwen3-8B_baseline/banking")
SAFE_AGENT_ROOT = Path("runs/Qwen_Qwen3-8B-safe-agent/banking")
OUTPUT_PATH = Path("runs/banking_security_improvements.txt")


def read_security_flag(path: Path) -> Optional[bool]:
    """Return the security flag from a JSON file, or None if unavailable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("security")
    except Exception as exc:  # pragma: no cover - defensive logging only
        print(f"Skipping {path}: {exc}")
        return None


def main() -> None:
    if not BASELINE_ROOT.is_dir():
        raise SystemExit(f"Missing baseline banking folder: {BASELINE_ROOT}")
    if not SAFE_AGENT_ROOT.is_dir():
        raise SystemExit(f"Missing safe-agent banking folder: {SAFE_AGENT_ROOT}")

    improvements = []

    for baseline_file in BASELINE_ROOT.rglob("*.json"):
        relative = baseline_file.relative_to(BASELINE_ROOT)
        safe_agent_file = SAFE_AGENT_ROOT / relative

        if not safe_agent_file.is_file():
            continue

        baseline_security = read_security_flag(baseline_file)
        safe_security = read_security_flag(safe_agent_file)

        if baseline_security is False and safe_security is True:
            improvements.append(baseline_file.as_posix())

    improvements.sort()
    OUTPUT_PATH.write_text("\n".join(improvements), encoding="utf-8")
    print(f"Found {len(improvements)} improvements. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
