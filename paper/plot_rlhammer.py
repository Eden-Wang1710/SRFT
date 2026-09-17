#!/usr/bin/env python3
"""RL-Hammer figures for the ICLR version: ASR vs attacker training epoch.

Two figures, each averaging TWO independent attacker runs per target and shading the run-to-run range:
  fig_rlhammer_llama.pdf  base Llama-3.1-8B / Meta-SecAlign-8B / SR-Agent-Llama
  fig_rlhammer_qwen.pdf   Qwen3-8B / SR-Agent-Qwen3-8B

Run dirs are those recorded in docs/13 §1b and §2d. SR-Agent-Llama run 1 was evaluated across two clusters
(ckpt 51-765 on skipjack, 816-1020 on WashU), so its early half is read straight out of the
`origin/exp/rlh-sr-llama` git tree -- see `read_series`.

    conda activate agentdojo && python paper/plot_rlhammer.py
"""
import json, re, subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "injecAgent-rl-harmmer" / "rl-injector" / "outputs"
STEPS_PER_EPOCH = 51

COLORS = {"base": "#D55E00", "secalign": "#009E73", "sr": "#0072B2"}
MARKERS = {"base": "s", "secalign": "D", "sr": "o"}


def read_dir(name):
    """checkpoint -> ASR (fraction) from outputs/<name>/*checkpoint-N*/attack_success_rate.json"""
    pts, base = {}, OUT / name
    if not base.exists():
        return pts
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)", child.name)
        if not m:
            continue
        for cand in (child / "attack_success_rate.json",
                     child / child.name / "attack_success_rate.json",
                     *sorted(child.glob("*/attack_success_rate.json"))):
            if cand.exists():
                pts[int(m.group(1))] = float(next(iter(json.load(cand.open()).values())))
                break
    return pts


def read_from_git(ref, name):
    """Same, for a run dir that lives on another branch (SR-Agent-Llama run 1, ckpt 51-765)."""
    rel = f"injecAgent-rl-harmmer/rl-injector/outputs/{name}/"
    try:
        listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, rel],
                                 cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except subprocess.CalledProcessError:
        return {}
    pts = {}
    for path in listing:
        if not path.endswith("attack_success_rate.json"):
            continue
        m = re.search(r"checkpoint-(\d+)", path)
        if not m:
            continue
        blob = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout
        pts[int(m.group(1))] = float(next(iter(json.loads(blob).values())))
    return pts


def series(spec):
    """spec: dir name, or (dir name, git ref) to merge an on-disk tail with a tail from another branch."""
    if isinstance(spec, tuple):
        name, ref = spec
        merged = read_from_git(ref, name)
        merged.update(read_dir(name))          # on-disk wins where both exist
        return merged
    return read_dir(spec)


def mean_band(run_a, run_b):
    """Mean of the two runs on their common checkpoints, plus the per-checkpoint min/max."""
    common = sorted(set(run_a) & set(run_b))
    ep = [c / STEPS_PER_EPOCH for c in common]
    mean = [100 * (run_a[c] + run_b[c]) / 2 for c in common]
    lo = [100 * min(run_a[c], run_b[c]) for c in common]
    hi = [100 * max(run_a[c], run_b[c]) for c in common]
    return ep, mean, lo, hi


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.9,
        "lines.markersize": 4.6,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def draw(curves, path, ymax=None, legend=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, key, (a, b) in curves:
        ra, rb = series(a), series(b)
        if not ra or not rb:
            raise SystemExit(f"missing data for {label}: {a} / {b}")
        ep, mean, lo, hi = mean_band(ra, rb)
        ax.fill_between(ep, lo, hi, color=COLORS[key], alpha=0.13, linewidth=0)
        ax.plot(ep, mean, color=COLORS[key], marker=MARKERS[key], label=label,
                markerfacecolor="white", markeredgewidth=1.3, clip_on=False, zorder=3)
        print(f"  {label:<26} epochs {ep[0]:.0f}-{ep[-1]:.0f}  final mean ASR {mean[-1]:.1f}%  peak {max(mean):.1f}%")
    ax.set_xlabel("Attacker training epoch")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, ymax or 100)
    ax.set_xticks(range(0, 21, 4))
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ncol = legend if legend else len(curves)
    leg = ax.legend(frameon=False, handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
                    loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=ncol)
    leg.get_frame().set_linewidth(0.6)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    style()
    print("Llama family:")
    draw([("Llama-3.1-8B-Instruct", "base", ("iclr_rlh_llama_base_", "iclr_rlh_llama_base_r3_")),
          ("Meta-SecAlign-8B", "secalign", ("eval_rl_hammer_target_meta_secalign_8b_allckpts_",
                                            "eval_rl_hammer_target_meta_secalign_8b_allckpts_rerun2_")),
          ("SR-Agent-Llama (ours)", "sr", (("iclr_rlh_srllama_noappend_", "origin/exp/rlh-sr-llama"),
                                           "iclr_rlh_srllama_noappend_r2_"))],
         ROOT / "paper" / "fig_rlhammer_llama.pdf")
    print("Qwen family:")
    draw([("Qwen3-8B", "base", ("eval_rl_hammer_target_llama_qwen_20epoch_YYN_allckpts_",
                                "eval_rl_hammer_target_llama_qwen_20epoch_YYN_rrreun_allckpts_")),
          ("SR-Agent-Qwen3-8B (ours)", "sr", ("YYY1", "YYY2"))],
         ROOT / "paper" / "fig_rlhammer_qwen.pdf")


if __name__ == "__main__":
    main()
