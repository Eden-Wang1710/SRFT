#!/usr/bin/env python3
"""Train-time ablation figure: RL-Hammer ASR vs attacker epoch on the Qwen3-8B family.

Four curves: the undefended base and full SRFT are the mean of two independent attacker runs; the two ablation arms are
single runs of the SAME checkpoint, differing only in the target's think mode, which is held consistent between attacker
training and evaluation. No run-to-run band is drawn (user preference, 2026-09-18).

    conda activate agentdojo && python paper/plot_ablation.py
"""
import json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "injecAgent-rl-harmmer" / "rl-injector" / "outputs"
STEPS_PER_EPOCH = 51


def series(name):
    pts, base = {}, OUT / name
    if not base.exists():
        raise SystemExit(f"missing run dir: {base}")
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        m = re.search(r"checkpoint-(\d+)", child.name)
        if not m:
            continue
        for cand in (child / "attack_success_rate.json", *sorted(child.glob("*/attack_success_rate.json"))):
            if cand.exists():
                pts[int(m.group(1))] = 100 * float(next(iter(json.load(cand.open()).values())))
                break
    return dict(sorted(pts.items()))


def mean_band(names):
    runs = [series(n) for n in names]
    common = sorted(set.intersection(*(set(r) for r in runs)))
    ep = [c / STEPS_PER_EPOCH for c in common]
    mean = [sum(r[c] for r in runs) / len(runs) for c in common]
    lo = [min(r[c] for r in runs) for c in common]
    hi = [max(r[c] for r in runs) for c in common]
    return ep, mean, lo, hi


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
        "axes.labelsize": 12, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.9, "lines.markersize": 4.6, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


CURVES = [
    ("Undefended base", "#D55E00", "s", "-",
     ["eval_rl_hammer_target_llama_qwen_20epoch_YYN_allckpts_",
      "eval_rl_hammer_target_llama_qwen_20epoch_YYN_rrreun_allckpts_"]),
    ("SR-Agent (ours)", "#0072B2", "o", "-", ["YYY1", "YYY2"]),
    ("w/o reflection, think on", "#CC79A7", "^", "-", ["iclr_rlh_ablq8_yyn_"]),
    ("w/o reflection, think off", "#CC79A7", "v", "--", ["iclr_rlh_ablq8_ynn_"]),
]
# --v2 (2026-09-23): adds the "w/o failure experience" arm (ABL-Q8-NOLAST, docs/01: paragraph 3 of the reflection removed,
# think + append on as for SR-Agent) and writes fig_ablation_v2.pdf; the original figure/list are left untouched.
# Add "iclr_rlh_ablq8_nolast_yyy_r2_" to the run list once run 2 (seed 2048) has been evaluated.
NOLAST_RUNS = ["iclr_rlh_ablq8_nolast_yyy_"]
CURVES_V2 = CURVES[:2] + [("w/o failure experience", "#009E73", "D", "-", NOLAST_RUNS)] + CURVES[2:]
V2 = "--v2" in sys.argv


def main():
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for label, color, marker, ls, names in (CURVES_V2 if V2 else CURVES):
        ep, mean, lo, hi = mean_band(names)
        ax.plot(ep, mean, color=color, marker=marker, linestyle=ls, label=label,
                markerfacecolor="white", markeredgewidth=1.2, clip_on=False, zorder=3)
        print(f"  {label:<26} epochs {ep[0]:.0f}-{ep[-1]:.0f}  final {mean[-1]:.1f}  peak {max(mean):.1f}")
    ax.set_xlabel("Attacker training epoch")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 80)
    ax.set_xticks(range(0, 21, 4))
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=2.2, columnspacing=1.6, handletextpad=0.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3 if V2 else 2)
    fig.subplots_adjust(bottom=0.34)
    out = ROOT / "paper" / ("fig_ablation_v2.pdf" if V2 else "fig_ablation.pdf")
    fig.savefig(out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
