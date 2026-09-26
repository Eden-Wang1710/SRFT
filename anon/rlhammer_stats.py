#!/usr/bin/env python3
"""RL-Hammer adaptive-attack results: attack success rate (ASR) versus attacker training epoch.

Reads   outputs/<target>/<run>/checkpoint-<step>/attack_success_rate.json
Writes  outputs/fig_rlhammer.pdf / .png   (Figure 3: (a) Llama-3.1-8B family, (b) Qwen3-8B family, (c) train-time ablation)
        outputs/rlhammer_summary.md       (Tables 7 and 8: Final / Last-5 / Peak (epoch), plus the per-run endpoints)

Conventions (paper Sec. 5.4, App. C.2): 51 attacker steps = 1 epoch, 20 epochs = 1020 steps; every checkpoint is
evaluated on the same 100 InjecAgent test cases. A target with two attacker runs is drawn as the mean over the two
runs on the checkpoints both runs have, with the run-to-run min/max shaded. Single-run ablation arms are drawn as is.

    pip install matplotlib   # only needed for the figure; the tables are written without it
    python rlhammer_stats.py
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STEPS_PER_EPOCH = 51

# label, colour key, [run dirs]   -- two runs = mean + band, one run = the run itself
PANELS = [
    ("(a) Llama-3.1-8B", [
        ("Undefended base",  "base",     ["llama31_8b_base/run1", "llama31_8b_base/run2"]),
        ("Meta-SecAlign-8B", "secalign", ["meta_secalign_8b/run1", "meta_secalign_8b/run2"]),
        ("SR-Agent (ours)",  "sr",       ["sr_agent_llama31_8b/run1", "sr_agent_llama31_8b/run2"]),
    ]),
    ("(b) Qwen3-8B", [
        ("Undefended base",  "base",     ["qwen3_8b_base/run1", "qwen3_8b_base/run2"]),
        ("SR-Agent (ours)",  "sr",       ["sr_agent_qwen3_8b/run1", "sr_agent_qwen3_8b/run2"]),
    ]),
    ("(c) Ablation on Qwen3-8B", [
        ("Undefended base",            "base", ["qwen3_8b_base/run1", "qwen3_8b_base/run2"]),
        ("SR-Agent (ours)",            "sr",   ["sr_agent_qwen3_8b/run1", "sr_agent_qwen3_8b/run2"]),
        ("w/o failure exp.",           "abl1", ["ablation_qwen3_8b/wo_failure_experience"]),
        ("w/o refl., think on",        "abl2", ["ablation_qwen3_8b/wo_reflection_think_on"]),
        ("w/o refl., think off",       "abl3", ["ablation_qwen3_8b/wo_reflection_think_off"]),
    ]),
]
COLORS = {"base": "#D55E00", "secalign": "#009E73", "sr": "#0072B2", "abl1": "#000000", "abl2": "#CC79A7", "abl3": "#CC79A7"}
MARKERS = {"base": "s", "secalign": "D", "sr": "o", "abl1": "D", "abl2": "^", "abl3": "v"}
STYLES = {"abl3": "--"}


def read_run(rel):
    """{step: ASR in %} for one run directory."""
    pts = {}
    for d in (OUT / rel).glob("checkpoint-*"):
        f = d / "attack_success_rate.json"
        m = re.search(r"checkpoint-(\d+)", d.name)
        if f.exists() and m:
            pts[int(m.group(1))] = 100 * float(next(iter(json.load(f.open()).values())))
    if not pts:
        sys.exit(f"no attack_success_rate.json under outputs/{rel}")
    return pts


def curve(runs):
    """epochs, mean, lo, hi, per-run values on the checkpoints common to all runs."""
    series = [read_run(r) for r in runs]
    common = sorted(set.intersection(*(set(s) for s in series)))
    ep = [c / STEPS_PER_EPOCH for c in common]
    per_run = [[s[c] for c in common] for s in series]
    mean = [sum(v) / len(v) for v in zip(*per_run)]
    lo = [min(v) for v in zip(*per_run)]
    hi = [max(v) for v in zip(*per_run)]
    return ep, mean, lo, hi, per_run


def summarize(label, runs):
    ep, mean, lo, hi, per_run = curve(runs)
    peak = max(mean)
    return {
        "target": label, "runs": len(runs), "epochs": f"{ep[0]:.0f}-{ep[-1]:.0f}",
        "final": mean[-1], "final_per_run": [v[-1] for v in per_run],
        "last5": sum(mean[-5:]) / len(mean[-5:]), "peak": peak, "peak_epoch": ep[mean.index(peak)],
    }


def table(title, rows):
    L = [f"### {title}", "", "| Target | Runs | Epochs | Final | per run | Last-5 | Peak (epoch) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['target']} | {r['runs']} | {r['epochs']} | {r['final']:.1f} | "
                 f"{' / '.join(f'{v:.0f}' for v in r['final_per_run'])} | {r['last5']:.1f} | {r['peak']:.1f} ({r['peak_epoch']:.0f}) |")
    return "\n".join(L) + "\n"


def main():
    md = ["# RL-Hammer endpoint statistics (ASR %, 100 InjecAgent test cases per checkpoint)", "",
          "Final = last common attacker epoch (mean over runs); Last-5 = mean over the last five epochs; "
          "Peak = maximum of the (mean) curve and the epoch at which it occurs.", ""]
    md.append(table("Table 7: main targets", [summarize(l, r) for _, curves in PANELS[:2] for l, _, r in curves]))
    md.append(table("Table 8: train-time ablation (Qwen3-8B)", [summarize(l, r) for l, _, r in PANELS[2][1]]))
    (OUT / "rlhammer_summary.md").write_text("\n".join(md))
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: tables written, figure skipped")
        return
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.8, "lines.markersize": 4.2, "figure.dpi": 150,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, (title, curves) in zip(axes, PANELS):
        for label, key, runs in curves:
            ep, mean, lo, hi, _ = curve(runs)
            if len(runs) > 1:
                ax.fill_between(ep, lo, hi, color=COLORS[key], alpha=0.13, linewidth=0)
            ax.plot(ep, mean, color=COLORS[key], marker=MARKERS[key], linestyle=STYLES.get(key, "-"), label=label,
                    markerfacecolor="white", markeredgewidth=1.2, clip_on=False, zorder=3)
        ax.set_title(title); ax.set_xlabel("Attacker training epoch")
        ax.set_xlim(0, 20.5); ax.set_ylim(0, 100); ax.set_xticks(range(0, 21, 4))
        ax.grid(axis="y", color="0.88", linewidth=0.7); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper left" if "Llama" in title else "upper left")
    axes[0].set_ylabel("Attack success rate (%)")
    fig.subplots_adjust(wspace=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_rlhammer.{ext}")
    print(f"figure -> {OUT / 'fig_rlhammer.pdf'}")


if __name__ == "__main__":
    main()
