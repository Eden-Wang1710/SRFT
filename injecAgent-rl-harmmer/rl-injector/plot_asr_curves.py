import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STEP_PER_EPOCH = 51
DEFAULT_EPOCHS = 20
PAPER_COLORS = {
    "SR-Agent": "#0072B2",
    "Qwen3-8B": "#D55E00",
    "Meta-SecAlign": "#009E73",
}
PAPER_MARKERS = {
    "SR-Agent": "o",
    "Qwen3-8B": "s",
    "Meta-SecAlign": "D",
}


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_safe_dirs = (
        repo_root / "outputs" / "YYY1",
        repo_root / "outputs" / "YYY2",
    )
    default_qwen_dirs = (
        repo_root / "outputs" / "eval_rl_hammer_target_llama_qwen_20epoch_YYN_allckpts_",
        repo_root / "outputs" / "eval_rl_hammer_target_llama_qwen_20epoch_YYN_rrreun_allckpts_",
    )
    default_meta_secalign_dirs = (
        repo_root / "outputs" / "eval_rl_hammer_target_meta_secalign_8b_allckpts_",
        repo_root / "outputs" / "eval_rl_hammer_target_meta_secalign_8b_allckpts_rerun2_",
    )
    default_output = repo_root / "outputs" / "asr_vs_epoch.png"

    parser = argparse.ArgumentParser(
        description="Plot ASR vs epoch after averaging two eval runs for safe-agent-8b, qwen3-8b, and meta-secalign-8b checkpoints."
    )
    parser.add_argument(
        "--safe-dirs",
        type=Path,
        nargs=2,
        default=default_safe_dirs,
        help="Two directories containing safe-agent-8b eval runs.",
    )
    parser.add_argument(
        "--qwen-dirs",
        type=Path,
        nargs=2,
        default=default_qwen_dirs,
        help="Two directories containing qwen3-8b eval runs.",
    )
    parser.add_argument(
        "--meta-secalign-dirs",
        type=Path,
        nargs=2,
        default=default_meta_secalign_dirs,
        help="Two directories containing meta-secalign-8b eval runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output image path.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=float,
        default=DEFAULT_STEP_PER_EPOCH,
        help="Checkpoint step divisor used to convert checkpoint-xx into epoch.",
    )
    return parser.parse_args()


def extract_checkpoint(folder_name: str):
    match = re.search(r"checkpoint-(\d+)", folder_name)
    return int(match.group(1)) if match else None


def read_asr(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not payload:
        raise ValueError(f"Empty ASR file: {json_path}")

    if len(payload) != 1:
        raise ValueError(f"Expected exactly one ASR entry in {json_path}, got {len(payload)}")

    return float(next(iter(payload.values())))


def find_asr_json(checkpoint_dir: Path):
    candidates = [
        checkpoint_dir / "attack_success_rate.json",
        checkpoint_dir / checkpoint_dir.name / "attack_success_rate.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(checkpoint_dir.glob("*/attack_success_rate.json"))
    return matches[0] if matches else None


def collect_series(base_dir: Path, label: str, required_substring: str):
    if not base_dir.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {base_dir}")

    points = {}
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if required_substring not in child.name:
            continue

        checkpoint = extract_checkpoint(child.name)
        if checkpoint is None:
            continue

        json_path = find_asr_json(child)
        if json_path is None:
            continue

        points[checkpoint] = read_asr(json_path)

    if not points:
        raise ValueError(f"No checkpointed ASR results found under {base_dir}")

    return sorted(points.items())


def fill_missing_edge_epochs(points, label: str, steps_per_epoch: float, epochs: int):
    by_checkpoint = dict(points)
    first_checkpoint = int(steps_per_epoch)
    second_checkpoint = int(steps_per_epoch * 2)
    last_checkpoint = int(steps_per_epoch * epochs)
    prev_checkpoint = int(steps_per_epoch * (epochs - 1))

    if first_checkpoint not in by_checkpoint and second_checkpoint in by_checkpoint:
        by_checkpoint[first_checkpoint] = by_checkpoint[second_checkpoint]
        print(
            f"Warning: {label} missing checkpoint-{first_checkpoint}; "
            f"using checkpoint-{second_checkpoint}."
        )

    if last_checkpoint not in by_checkpoint:
        replacement_checkpoint = prev_checkpoint
        if replacement_checkpoint not in by_checkpoint:
            earlier_checkpoints = [
                checkpoint for checkpoint in by_checkpoint if checkpoint < last_checkpoint
            ]
            if not earlier_checkpoints:
                raise ValueError(
                    f"{label} missing checkpoint-{last_checkpoint} and has no earlier "
                    "checkpoint to use as replacement."
                )
            replacement_checkpoint = max(earlier_checkpoints)
            print(
                f"Warning: {label} missing checkpoint-{prev_checkpoint}; "
                f"using checkpoint-{replacement_checkpoint} for checkpoint-{last_checkpoint}."
            )
        else:
            print(
                f"Warning: {label} missing checkpoint-{last_checkpoint}; "
                f"using checkpoint-{replacement_checkpoint}."
            )

        by_checkpoint[last_checkpoint] = by_checkpoint[replacement_checkpoint]

    return sorted(by_checkpoint.items())


def average_eval_runs(run_points, label: str, steps_per_epoch: float):
    run_maps = [dict(points) for points in run_points]
    common_checkpoints = sorted(set(run_maps[0]).intersection(run_maps[1]))
    if not common_checkpoints:
        raise ValueError(f"No overlapping checkpoints found for {label}")

    missing_by_run = [
        sorted(set(run_maps[1 - idx]) - set(run_maps[idx])) for idx in range(len(run_maps))
    ]
    for idx, missing_checkpoints in enumerate(missing_by_run, start=1):
        if missing_checkpoints:
            missing_epochs = [checkpoint / steps_per_epoch for checkpoint in missing_checkpoints]
            print(f"Warning: {label} run {idx} not averaged for epochs: {missing_epochs}")

    return [
        (checkpoint, sum(run_map[checkpoint] for run_map in run_maps) / len(run_maps))
        for checkpoint in common_checkpoints
    ]


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_series(series_by_label, output_path: Path, steps_per_epoch: float):
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for label, points in series_by_label:
        epochs = [ckpt / steps_per_epoch for ckpt, _ in points]
        asr = [value for _, value in points]
        ax.plot(
            epochs,
            asr,
            color=PAPER_COLORS.get(label),
            marker=PAPER_MARKERS.get(label, "o"),
            linewidth=1.8,
            markersize=4.5,
            markeredgewidth=0.8,
            label=label,
        )

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Attack success rate (ASR)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.9, 20.25)
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.tight_layout(pad=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_output_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_output_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_output_path


def main():
    args = parse_args()
    series_by_label = []
    for label, dirs in (
        ("SR-Agent", args.safe_dirs),
        ("Qwen3-8B", args.qwen_dirs),
        ("Meta-SecAlign", args.meta_secalign_dirs),
    ):
        run_points = [
            fill_missing_edge_epochs(
                collect_series(run_dir, f"{label} run {idx}", "_checkpoint-"),
                f"{label} run {idx}",
                args.steps_per_epoch,
                DEFAULT_EPOCHS,
            )
            for idx, run_dir in enumerate(dirs, start=1)
        ]
        series_by_label.append(
            (label, average_eval_runs(run_points, label, args.steps_per_epoch))
        )

    pdf_output_path = plot_series(
        series_by_label,
        args.output,
        args.steps_per_epoch,
    )
    print(f"Saved plot to: {args.output}")
    print(f"Saved PDF to: {pdf_output_path}")


if __name__ == "__main__":
    main()
