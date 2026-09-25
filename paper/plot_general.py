#!/usr/bin/env python3
"""General-capability figure for the ICLR version (the counterpart of ReasAlign's Fig. 3): grouped bars for
MMLU / MMLU-Pro / IFEval / BBH / average, three models -- our base Llama-3.1-8B-Instruct, Meta-SecAlign-8B (the
numbers published in its paper, arXiv 2507.02735 Table 3) and SR-Agent-Llama.

Numbers are the FINAL rows of docs/13 §5 (settled 2026-09-16; every cell reproducible with eval_general/report.py):
  MMLU      Meta's 0-shot-CoT recipe (`meta_mmlu_0shot_instruct`), the same protocol as the published rows
  MMLU-Pro  5-shot CoT, official three-tier extraction; ours on 100 items/subject, SR with the CoT prefill
  IFEval    mean of the four sub-metrics (the published convention)
  BBH       3-shot CoT; ours with the tolerant answer extractor (base default run, SR relaxed-stop run)
Caveat for the caption: the Meta-SecAlign column is copied from its paper (their harness); our two columns share
one harness. Their base row was 72.0 / 46.5 / 79.1 / 71.9 -- ours reproduces it within 1.3 points everywhere.

    conda activate agentdojo && python paper/plot_general.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

BENCH = ["MMLU", "MMLU-Pro", "IFEval", "BBH"]
MODELS = [  # (key, legend label, scores in BENCH order)
    ("base",     "Llama-3.1-8B-Instruct",        [71.83, 47.50, 79.26, 73.17]),
    ("secalign", "Meta-SecAlign-8B",             [71.7,  46.7,  74.5,  70.9]),
    ("sr",       "SR-Agent-Llama (ours)",        [71.91, 45.79, 79.70, 71.17]),
]
# same series colours as paper/plot_rlhammer.py (Okabe-Ito, colour-blind safe), so the two figures read as one set
COLORS = {"base": "#D55E00", "secalign": "#009E73", "sr": "#0072B2"}
HATCH = {"base": "", "secalign": "//", "sr": ""}   # secondary encoding for print / CVD


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
        "axes.labelsize": 12, "legend.fontsize": 10, "xtick.labelsize": 10.5, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
        "hatch.linewidth": 0.6,
    })


def main():
    style()
    groups = BENCH + ["Average"]
    x = np.arange(len(groups))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 2.2))  # flat layout (2026-09-24); legend sits inside the axes above the bars
    for i, (key, label, vals) in enumerate(MODELS):
        vals = list(vals) + [float(np.mean(vals))]
        offs = (i - 1) * width
        bars = ax.bar(x + offs, vals, width * 0.92, label=label, color=COLORS[key], hatch=HATCH[key],
                      edgecolor="white", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.2,
                    color="#222222", zorder=4)
    ax.set_xticks(x, groups)
    ax.set_ylim(30, 97)
    ax.set_yticks([30, 50, 70])
    ax.set_ylabel("Score (%)")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.5,
              columnspacing=1.2, handletextpad=0.5, fontsize=9.5, borderaxespad=0)
    fig.savefig(HERE / "fig_general.pdf")
    fig.savefig(HERE / "fig_general.png", dpi=200)
    for key, label, vals in MODELS:
        print(f"{label:24s} " + " ".join(f"{v:6.2f}" for v in vals) + f"  avg {np.mean(vals):6.2f}")


if __name__ == "__main__":
    main()
