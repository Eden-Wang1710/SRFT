#!/usr/bin/env python3
"""
Compute the ratio of attacked samples where utility=True and safety/security=False.

The existing result JSON files in this repo use "security" rather than "safety",
so this script reads "safety" first and falls back to "security".
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def is_attacked(path, data):
    normalized = path.replace("\\", "/")
    if "/important_instructions/" in normalized:
        return True
    if isinstance(data, dict) and data.get("attack_type") == "important_instructions":
        return True
    return False


def collect_json_files(base_dir):
    out = []
    for root, _dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(".json"):
                out.append(os.path.join(root, filename))
    return out


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print(f"Warning: failed to load {path}: {exc}")
        return None


def safety_value(data):
    if not isinstance(data, dict):
        return None
    if "safety" in data:
        return data.get("safety")
    return data.get("security")


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def analyze(runs_root):
    scenarios = [
        os.path.join(runs_root, name)
        for name in os.listdir(runs_root)
        if os.path.isdir(os.path.join(runs_root, name))
    ]

    results = []
    global_total = 0
    global_matches = 0

    for scenario_path in sorted(scenarios):
        scenario = os.path.basename(scenario_path)
        counts = defaultdict(int)

        for path in collect_json_files(scenario_path):
            data = load_json(path)
            if not is_attacked(path, data):
                continue

            counts["attacked_total"] += 1
            utility = data.get("utility") if isinstance(data, dict) else None
            safety = safety_value(data)

            if utility is True and safety is False:
                counts["utility_true_safety_false"] += 1

        if counts["attacked_total"] == 0:
            continue

        global_total += counts["attacked_total"]
        global_matches += counts["utility_true_safety_false"]

        results.append(
            {
                "scenario": scenario,
                "attacked_total": counts["attacked_total"],
                "utility_true_safety_false": counts["utility_true_safety_false"],
                "utility_true_safety_false_rate": rate(
                    counts["utility_true_safety_false"], counts["attacked_total"]
                ),
            }
        )

    results.append(
        {
            "scenario": "ALL",
            "attacked_total": global_total,
            "utility_true_safety_false": global_matches,
            "utility_true_safety_false_rate": rate(global_matches, global_total),
        }
    )
    return results


def write_csv(results, out_csv):
    fieldnames = [
        "scenario",
        "attacked_total",
        "utility_true_safety_false",
        "utility_true_safety_false_rate",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            formatted = row.copy()
            value = formatted["utility_true_safety_false_rate"]
            formatted["utility_true_safety_false_rate"] = (
                f"{value:.6f}" if isinstance(value, float) else ""
            )
            writer.writerow(formatted)


def print_results(results):
    for row in results:
        print(f"Scenario: {row['scenario']}")
        print(
            "  Attacked: "
            f"total={row['attacked_total']}, "
            f"utility_true_safety_false={row['utility_true_safety_false']}, "
            f"rate={row['utility_true_safety_false_rate']}"
        )
        print("-" * 60)


def find_runs_root(run_name, runs_base):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if runs_base:
        candidates = [os.path.join(runs_base, run_name)]
    else:
        candidates = [
            os.path.join(script_dir, "..", "runs", run_name),
            os.path.join(script_dir, "..", "..", "runs", run_name),
            os.path.join(script_dir, "runs", run_name),
            os.path.join(os.getcwd(), "runs", run_name),
            os.path.join(
                "/data/xiaogeng_liu/experiments/zixuan/agentdojo/runs", run_name
            ),
        ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    print(f"Error: could not find runs/{run_name}. Tried locations: {candidates}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute utility=True and safety/security=False rate for attacked samples."
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default="Qwen_Qwen3-8B-safe-agent",
        help="Name of the run directory under runs/.",
    )
    parser.add_argument(
        "--runs-base",
        default=None,
        help="Optional base path that contains the run directories.",
    )
    args = parser.parse_args()

    runs_root = find_runs_root(args.run_name, args.runs_base)
    if not runs_root:
        return 2

    results = analyze(runs_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    safe_name = args.run_name.replace("/", "_")
    out_csv = os.path.join(script_dir, f"utility_safety_stats_{safe_name}.csv")

    write_csv(results, out_csv)
    print("\nComputed utility=True and safety/security=False metrics:")
    print_results(results)
    print(f"CSV written to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
