#!/usr/bin/env python3
"""Copy the RL-Hammer result directories behind Fig. 3 / Tables 7-8 into the anonymous repo with a uniform layout.

    outputs/<target>/<run>/checkpoint-<step>/attack_success_rate.json   (always, when the source has it)
    outputs/<target>/<run>/checkpoint-<step>/test_cases.json            (per-case outputs, when the source has them)

SR-Agent-Llama run 1 was evaluated on two clusters: checkpoints 51-765 live only on the git branch
`origin/exp/rlh-sr-llama`, 816-1020 on disk. Both halves are merged into one series.
Files are copied verbatim (no rewriting of contents).
"""
import json, re, shutil, subprocess, sys
from pathlib import Path

SRC_ROOT = Path(sys.argv[1])            # SRFT repo root
DST = Path(sys.argv[2])                 # <anon>/injecAgent-rl-harmmer/rl-injector/outputs
OUT = SRC_ROOT / "injecAgent-rl-harmmer" / "rl-injector" / "outputs"
GIT_REF = "origin/exp/rlh-sr-llama"

# (target, run, source dir name, optional git ref holding additional checkpoints of the same run)
RUNS = [
    ("llama31_8b_base",            "run1", "iclr_rlh_llama_base_", None),
    ("llama31_8b_base",            "run2", "iclr_rlh_llama_base_r3_", None),
    ("meta_secalign_8b",           "run1", "eval_rl_hammer_target_meta_secalign_8b_allckpts_", None),
    ("meta_secalign_8b",           "run2", "eval_rl_hammer_target_meta_secalign_8b_allckpts_rerun2_", None),
    ("sr_agent_llama31_8b",        "run1", "iclr_rlh_srllama_noappend_", GIT_REF),
    ("sr_agent_llama31_8b",        "run2", "iclr_rlh_srllama_noappend_r2_", None),
    ("qwen3_8b_base",              "run1", "eval_rl_hammer_target_llama_qwen_20epoch_YYN_allckpts_", None),
    ("qwen3_8b_base",              "run2", "eval_rl_hammer_target_llama_qwen_20epoch_YYN_rrreun_allckpts_", None),
    ("sr_agent_qwen3_8b",          "run1", "YYY1", None),
    ("sr_agent_qwen3_8b",          "run2", "YYY2", None),
    ("ablation_qwen3_8b",          "wo_failure_experience",    "iclr_rlh_ablq8_nolast_yyy_", None),
    ("ablation_qwen3_8b",          "wo_reflection_think_on",   "iclr_rlh_ablq8_yyn_", None),
    ("ablation_qwen3_8b",          "wo_reflection_think_off",  "iclr_rlh_ablq8_ynn_", None),
]
STEP_RE = re.compile(r"checkpoint-(\d+)")


def classify(name):
    if name == "attack_success_rate.json":
        return "asr"
    if name.endswith(".json"):
        return "cases"
    return None


def from_disk(src_dir):
    """{step: {'asr': Path, 'cases': Path}} — the first file found per kind under each checkpoint dir."""
    found = {}
    for child in sorted(src_dir.iterdir()):
        m = STEP_RE.search(child.name)
        if not child.is_dir() or not m:
            continue
        step = int(m.group(1))
        for f in sorted(child.rglob("*.json")):
            kind = classify(f.name)
            if kind and kind not in found.setdefault(step, {}):
                found[step][kind] = f
    return found


def from_git(ref, name):
    rel = f"injecAgent-rl-harmmer/rl-injector/outputs/{name}/"
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, rel], cwd=SRC_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    found = {}
    for path in listing:
        m = STEP_RE.search(path)
        kind = classify(Path(path).name)
        if not m or not kind:
            continue
        found.setdefault(int(m.group(1)), {}).setdefault(kind, ("git", path))
    return found


def write(dst_file, src):
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(src, tuple):
        blob = subprocess.run(["git", "show", f"{GIT_REF}:{src[1]}"], cwd=SRC_ROOT,
                              capture_output=True, check=True).stdout
        dst_file.write_bytes(blob)
    else:
        shutil.copyfile(src, dst_file)


summary = []
for target, run, name, ref in RUNS:
    src_dir = OUT / name
    if not src_dir.is_dir():
        sys.exit(f"missing source run dir: {src_dir}")
    found = {}
    if ref:
        found.update(from_git(ref, name))
    for step, files in from_disk(src_dir).items():      # on-disk wins where both exist
        found.setdefault(step, {}).update(files)
    n_asr = n_cases = 0
    for step, files in sorted(found.items()):
        d = DST / target / run / f"checkpoint-{step}"
        if "asr" in files:
            write(d / "attack_success_rate.json", files["asr"]); n_asr += 1
        if "cases" in files:
            write(d / "test_cases.json", files["cases"]); n_cases += 1
    summary.append((target, run, name, len(found), n_asr, n_cases))

print(f"{'target':22s} {'run':26s} {'source':64s} ckpts asr cases")
for t, r, n, k, a, c in summary:
    print(f"{t:22s} {r:26s} {n:64s} {k:5d} {a:3d} {c:5d}")
(DST / "SOURCES.json").write_text(json.dumps(
    [{"target": t, "run": r, "checkpoints": k, "asr_files": a, "per_case_files": c} for t, r, n, k, a, c in summary],
    indent=1) + "\n")
