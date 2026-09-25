#!/usr/bin/env python3
"""One-row figure for the ICLR main text (2026-09-24): RL-Hammer curves for (a) Llama-3.1-8B, (b) Qwen3-8B, and (c) the
train-time ablation on Qwen3-8B, sharing one y axis and one legend. Replaces fig_rlhammer.pdf + fig_ablation_v2.pdf in the
main text; the data, colours and markers are exactly those of plot_rlhammer.py / plot_ablation.py (imported below).

    conda activate agentdojo && python paper/plot_rlhammer_ablation.py
The ablation run dirs (iclr_rlh_ablq8_*) are tracked on exp/ablation only; if they are not in this checkout, point
SRFT_ABL_OUT at an outputs/ dir that has them (e.g. the shared tree's).
"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_rlhammer as R
import plot_ablation as A

if not (A.OUT / A.NOLAST_RUNS[0]).exists() and os.environ.get("SRFT_ABL_OUT"):
    A.OUT = Path(os.environ["SRFT_ABL_OUT"])

PANELS_AB = [
    ("(a) Llama-3.1-8B",
     [("Undefended base", "base", ("iclr_rlh_llama_base_", "iclr_rlh_llama_base_r3_")),
      ("Meta-SecAlign-8B", "secalign", ("eval_rl_hammer_target_meta_secalign_8b_allckpts_",
                                        "eval_rl_hammer_target_meta_secalign_8b_allckpts_rerun2_")),
      ("SR-Agent (ours)", "sr", (("iclr_rlh_srllama_noappend_", "origin/exp/rlh-sr-llama"),
                                 "iclr_rlh_srllama_noappend_r2_"))]),
    ("(b) Qwen3-8B",
     [("Undefended base", "base", ("eval_rl_hammer_target_llama_qwen_20epoch_YYN_allckpts_",
                                   "eval_rl_hammer_target_llama_qwen_20epoch_YYN_rrreun_allckpts_")),
      ("SR-Agent (ours)", "sr", ("YYY1", "YYY2"))]),
]
# legend is filled column-major (ncol=3): this order gives row 1 = the three main-text models, row 2 = the three ablation arms
ORDER = ["Undefended base", "w/o failure experience", "Meta-SecAlign-8B", "w/o reflection, think on",
         "SR-Agent (ours)", "w/o reflection, think off"]


def main():
    R.style()
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "lines.linewidth": 1.7, "lines.markersize": 4.0})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    handles = {}
    for ax, (title, curves) in zip(axes[:2], PANELS_AB):
        for label, key, (a, b) in curves:
            ra, rb = R.series(a), R.series(b)
            if not ra or not rb:
                raise SystemExit(f"missing data for {label}: {a} / {b}")
            ep, mean, lo, hi = R.mean_band(ra, rb)
            ax.fill_between(ep, lo, hi, color=R.COLORS[key], alpha=0.13, linewidth=0)
            line, = ax.plot(ep, mean, color=R.COLORS[key], marker=R.MARKERS[key], label=label,
                            markerfacecolor="white", markeredgewidth=1.1, clip_on=False, zorder=3)
            handles.setdefault(label, line)
            print(f"  {title:<18} {label:<26} final {mean[-1]:.1f}  peak {max(mean):.1f}")
        ax.set_title(title, pad=6)
    ax = axes[2]
    for label, color, marker, ls, names in A.CURVES_V2:
        ep, mean, lo, hi = A.mean_band(names)
        line, = ax.plot(ep, mean, color=color, marker=marker, linestyle=ls, label=label,
                        markerfacecolor="white", markeredgewidth=1.1, clip_on=False, zorder=3)
        handles.setdefault(label, line)
        print(f"  (c) ablation       {label:<26} final {mean[-1]:.1f}  peak {max(mean):.1f}")
    ax.set_title("(c) Train-time ablation (Qwen3-8B)", pad=6)
    for ax in axes:
        ax.set_xlim(0, 20.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 21, 4))
        ax.set_yticks(range(0, 101, 25))
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("Attacker training epoch")
    axes[0].set_ylabel("ASR (%)")
    fig.legend([handles[k] for k in ORDER], ORDER, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               frameon=False, handlelength=1.8, columnspacing=2.4, handletextpad=0.5, labelspacing=0.3)
    fig.subplots_adjust(wspace=0.10, bottom=0.36, top=0.88, left=0.07, right=0.99)
    out = HERE / "fig_rlhammer_ablation.pdf"
    fig.savefig(out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
