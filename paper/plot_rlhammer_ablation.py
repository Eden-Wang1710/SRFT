#!/usr/bin/env python3
"""One-row figure for the ICLR main text (2026-09-24): RL-Hammer curves for (a) Llama-3.1-8B, (b) Qwen3-8B, and (c) the
train-time ablation on Qwen3-8B, sharing one y axis. Two legends: the three main-text models below the figure, the three
ablation arms inside panel (c). Replaces fig_rlhammer.pdf + fig_ablation_v2.pdf in the main text; the data are exactly
those of plot_rlhammer.py / plot_ablation.py (imported below), colours/markers restyled for the small panels.

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
# panel (c): SR-Agent (same runs/colour as in (b)) + the three ablation arms; the undefended base is left out (it is in (b)
# and hid the arms). Restyled: short labels, dark grey instead of black, purple vs pink instead of solid vs dashed pink.
ABL_STYLE = {
    "SR-Agent (ours)":           dict(label=None,                    color=R.COLORS["sr"], marker="o", ls="-"),
    "w/o failure experience":    dict(label="w/o failure exp.",      color="#555555",      marker="D", ls="-"),
    "w/o reflection, think on":  dict(label="w/o refl., think on",   color="#8E44AD",      marker="^", ls="-"),
    "w/o reflection, think off": dict(label="w/o refl., think off",  color="#CC79A7",      marker="v", ls="--"),
}
MAIN_ORDER = ["Undefended base", "Meta-SecAlign-8B", "SR-Agent (ours)"]
EVERY = 2  # marker every 2 epochs


def main():
    R.style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 11.5, "legend.fontsize": 10.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "lines.linewidth": 1.6, "lines.markersize": 4.2})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    handles = {}
    for ax, (title, curves) in zip(axes[:2], PANELS_AB):
        for label, key, (a, b) in curves:
            ra, rb = R.series(a), R.series(b)
            if not ra or not rb:
                raise SystemExit(f"missing data for {label}: {a} / {b}")
            ep, mean, lo, hi = R.mean_band(ra, rb)
            ax.fill_between(ep, lo, hi, color=R.COLORS[key], alpha=0.08, linewidth=0)
            line, = ax.plot(ep, mean, color=R.COLORS[key], marker=R.MARKERS[key], markevery=EVERY, label=label,
                            markerfacecolor="white", markeredgewidth=1.1, clip_on=False, zorder=3)
            handles.setdefault(label, line)
            print(f"  {title:<18} {label:<26} final {mean[-1]:.1f}  peak {max(mean):.1f}")
        ax.set_title(title, pad=6)
    ax = axes[2]
    abl_handles, abl_labels = [], []
    for label, _c, _m, _ls, names in A.CURVES_V2:
        if label not in ABL_STYLE:
            continue
        st = ABL_STYLE[label]
        ep, mean, lo, hi = A.mean_band(names)
        line, = ax.plot(ep, mean, color=st["color"], marker=st["marker"], linestyle=st["ls"], markevery=EVERY,
                        markerfacecolor="white", markeredgewidth=1.1, clip_on=False, zorder=3)
        if st["label"]:
            abl_handles.append(line); abl_labels.append(st["label"])
        print(f"  (c) ablation       {label:<26} final {mean[-1]:.1f}  peak {max(mean):.1f}")
    ax.set_title("(c) Ablation on Qwen3-8B", pad=6)
    ax.legend(abl_handles, abl_labels, loc="upper left", bbox_to_anchor=(-0.01, 1.03), frameon=False, fontsize=9.5,
              handlelength=1.7, handletextpad=0.5, labelspacing=0.2, borderaxespad=0)
    for ax in axes:
        ax.set_xlim(0, 20.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(0, 21, 4))
        ax.set_yticks(range(0, 101, 25))
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ASR (%)")
    fig.supxlabel("Attacker training epoch", y=0.115, fontsize=11.5)
    fig.legend([handles[k] for k in MAIN_ORDER], MAIN_ORDER, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=False, handlelength=1.8, columnspacing=2.6, handletextpad=0.5)
    fig.subplots_adjust(wspace=0.10, bottom=0.27, top=0.89, left=0.07, right=0.99)
    out = HERE / "fig_rlhammer_ablation.pdf"
    fig.savefig(out)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
