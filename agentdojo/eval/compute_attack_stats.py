#!/usr/bin/env python3
"""
Compute completing rates and attack metrics for runs/Qwen_Qwen3-8B-safe-agent.

Outputs:
 - prints per-scenario stats to stdout
 - writes CSV to eval/attack_stats.csv

Behavior notes / assumptions:
 - Files under a scenario path containing '/important_instructions/' are treated as attacked samples.
 - Files under a scenario path containing '/none/' are treated as clean user_task samples.
 - Files directly under a scenario with name or folder 'injection_task' or whose basename starts with 'injection_task_' are treated as "pure injection" samples.
 - If fields are missing, they are treated as None and ignored when counting that metric where applicable.
"""
import csv
import json
import os
from collections import defaultdict
import math
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    plt = None
    np = None


def is_attacked(path, data):
    if '/important_instructions/' in path.replace('\\', '/'):
        return True
    # fallback: some files contain attack_type in JSON
    if isinstance(data, dict) and data.get('attack_type') == 'important_instructions':
        return True
    return False


def is_none(path):
    return '/none/' in path.replace('\\', '/')


def is_pure_injection(path):
    p = path.replace('\\', '/')
    # a file or folder name at scenario root like 'injection_task_0.json' or 'injection_task_0/...'
    return ('/injection_task_' in p) and ('/user_task_' not in p)


def collect_json_files(base_dir):
    out = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            path = os.path.join(root, f)
            out.append(path)
    return out


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def analyze(runs_root):
    # Expect runs_root contains directories per scenario
    scenarios = [os.path.join(runs_root, d) for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))]

    results = []

    # global aggregates
    g_pure_total = 0
    g_pure_true = 0
    g_user_total = 0
    g_user_true = 0
    g_att_total = 0
    g_att_security_true = 0
    # utility under attack counts utility==True AND security==False
    g_att_utility_true_security_false = 0

    for scen_path in sorted(scenarios):
        scen_name = os.path.basename(scen_path)
        files = collect_json_files(scen_path)
        if not files:
            continue

        stats = defaultdict(int)
        totals = defaultdict(int)

        # Counters for groups
        pure_injection_total = 0
        pure_injection_utility_true = 0

        user_clean_total = 0
        user_clean_utility_true = 0

        attacked_total = 0
        # per-scenario count of security==True (attack succeeded)
        attacked_security_true = 0
        # utility under attack: utility==True and security==False
        attacked_utility_true_security_false = 0

        for p in files:
            data = load_json(p)
            utility = None
            security = None
            if isinstance(data, dict):
                utility = data.get('utility')
                security = data.get('security')

            attacked = is_attacked(p, data)
            pure_inj = is_pure_injection(p)
            # Exclude pure injection files from being counted as user 'none' samples
            clean_user = (not attacked) and is_none(p) and (not pure_inj)

            # Pure injection
            if pure_inj:
                pure_injection_total += 1
                if utility is True:
                    pure_injection_utility_true += 1

            # add to global pure
            if pure_inj:
                g_pure_total += 1
                if utility is True:
                    g_pure_true += 1

            # user_task clean (none)
            if clean_user:
                user_clean_total += 1
                if utility is True:
                    user_clean_utility_true += 1

            # add to global user clean
            if clean_user:
                g_user_total += 1
                if utility is True:
                    g_user_true += 1

            # attacked samples
            if attacked:
                attacked_total += 1
                # count attack success as security == True
                if security is True:
                    attacked_security_true += 1
                # utility under attack: solved correctly without adversarial side-effects -> utility==True and security==False
                if utility is True :
                    attacked_utility_true_security_false += 1

            # add to global attacked
            if attacked:
                g_att_total += 1
                if security is True:
                    g_att_security_true += 1
                if utility is True :
                    g_att_utility_true_security_false += 1

        # prepare metrics (handle zero denominators)
        def rate(n, d):
            return (n / d) if d else None

        results.append({
            'scenario': scen_name,
            'pure_injection_total': pure_injection_total,
            'pure_injection_completing_rate': rate(pure_injection_utility_true, pure_injection_total),
            'user_clean_total': user_clean_total,
            'user_clean_completing_rate': rate(user_clean_utility_true, user_clean_total),
            'attacked_total': attacked_total,
            'attacked_ASR': rate(attacked_security_true, attacked_total),
            'utility_under_attack_rate': rate(attacked_utility_true_security_false, attacked_total),
        })

    # append aggregated 'ALL' row (computed from summed counts to avoid averaging rates)
    results.append({
        'scenario': 'ALL',
        'pure_injection_total': g_pure_total,
        'pure_injection_completing_rate': rate(g_pure_true, g_pure_total),
        'user_clean_total': g_user_total,
        'user_clean_completing_rate': rate(g_user_true, g_user_total),
        'attacked_total': g_att_total,
        'attacked_ASR': rate(g_att_security_true, g_att_total),
        'utility_under_attack_rate': rate(g_att_utility_true_security_false, g_att_total),
    })

    return results


def write_csv(results, out_csv):
    fieldnames = [
        'scenario',
        'user_clean_total',
        'user_clean_completing_rate',
        'pure_injection_total',
        'pure_injection_completing_rate',
        'attacked_total',
        'utility_under_attack_rate',
        'attacked_ASR',
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            # format rates as floats or empty
            row = r.copy()
            for k in ['pure_injection_completing_rate', 'user_clean_completing_rate', 'attacked_ASR', 'utility_under_attack_rate']:
                v = row.get(k)
                row[k] = f"{v:.6f}" if isinstance(v, float) else ('' if v is None else str(v))
            w.writerow(row)


def print_results(results):
    for r in results:
        print(f"Scenario: {r['scenario']}")
        print(f"  Clean utility: total={r['user_clean_total']}, completing_rate={r['user_clean_completing_rate']}")
        print(f"  Pure injection: total={r['pure_injection_total']}, completing_rate={r['pure_injection_completing_rate']}")
        print(f"  Attacked: total={r['attacked_total']}, UtilityUnderAttack={r['utility_under_attack_rate']}, ASR={r['attacked_ASR']}")
        print('-' * 60)


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(description='Compute attack stats for a runs/<run_name> directory')
    parser.add_argument('run_name', nargs='?', default='Qwen_Qwen3-8B-safe-agent',
                        help='Name of the run directory under runs/, e.g. Qwen_Qwen3-8B-safe-agent')
    parser.add_argument('--runs-base', default=None,
                        help='Optional base path that contains the run directories. If omitted, a few sensible locations relative to the script and cwd will be tried.')
    args = parser.parse_args()

    run_name = args.run_name
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # candidates: if runs_base provided, use it; otherwise try several common locations
    if args.runs_base:
        candidates = [os.path.join(args.runs_base, run_name)]
    else:
        candidates = [
            os.path.join(script_dir, '..', 'runs', run_name),
            os.path.join(script_dir, '..', '..', 'runs', run_name),
            os.path.join(script_dir, 'runs', run_name),
            os.path.join(os.getcwd(), 'runs', run_name),
            os.path.join('/data/xiaogeng_liu/experiments/zixuan/agentdojo/runs', run_name),
        ]

    runs_root = None
    for c in candidates:
        if os.path.isdir(c):
            runs_root = os.path.abspath(c)
            break

    if not runs_root:
        print(f"Error: could not find runs/{run_name}. Tried locations: {candidates}")
        return 2

    results = analyze(runs_root)
    safe_name = run_name.replace('/', '_')
    out_csv = os.path.join(script_dir, f'attack_stats_{safe_name}.csv')

    write_csv(results, out_csv)
    print('\nComputed metrics:')
    print_results(results)
    print(f'CSV written to: {out_csv}')

    # Attempt to plot results
    out_png = os.path.join(script_dir, f'attack_stats_{safe_name}.png')
    if plt is None or np is None:
        print('Matplotlib or numpy not available; skipping plotting. To enable plotting: pip install matplotlib numpy')
    else:
        try:
            # prepare arrays
            scenarios = [r['scenario'] for r in results]
            def val_or_nan(x):
                return float('nan') if x is None else float(x)

            pure_rates = [val_or_nan(r.get('pure_injection_completing_rate')) for r in results]
            user_rates = [val_or_nan(r.get('user_clean_completing_rate')) for r in results]
            asr_rates = [val_or_nan(r.get('attacked_ASR')) for r in results]
            uua_rates = [val_or_nan(r.get('utility_under_attack_rate')) for r in results]

            x = np.arange(len(scenarios))
            width = 0.2

            fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 5))
            # Order: Clean utility, Pure injection, UtilityUnderAttack, ASR
            ax.bar(x - 1.5 * width, user_rates, width, label='Clean utility completing rate')
            ax.bar(x - 0.5 * width, pure_rates, width, label='Pure injection completing rate')
            ax.bar(x + 0.5 * width, uua_rates, width, label='Utility under attack')
            ax.bar(x + 1.5 * width, asr_rates, width, label='Attacked ASR')

            ax.set_ylabel('Rate (0-1)')
            ax.set_title('Attack statistics by scenario')
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_png)
            print(f'Plot saved to: {out_png}')
        except Exception as e:
            print(f'Warning: failed to create plot: {e}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
