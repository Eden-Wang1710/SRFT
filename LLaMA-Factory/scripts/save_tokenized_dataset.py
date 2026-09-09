#!/usr/bin/env python3
"""Save a tokenized dataset without launching model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.misc import has_tokenized_data
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize a LLaMA-Factory training dataset.")
    parser.add_argument("config", type=Path, help="Training YAML config.")
    parser.add_argument("tokenized_path", type=Path, help="Output path for the tokenized dataset.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing tokenized dataset.",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def main() -> None:
    args = parse_args()
    if has_tokenized_data(str(args.tokenized_path)) and not args.overwrite:
        print(f"Tokenized dataset already exists: {args.tokenized_path}")
        return

    config = OmegaConf.load(args.config)
    overrides = OmegaConf.merge(
        OmegaConf.from_cli(args.overrides),
        {
            "tokenized_path": str(args.tokenized_path),
            "overwrite_cache": True,
            "do_train": True,
        },
    )
    merged_config = OmegaConf.to_container(OmegaConf.merge(config, overrides), resolve=True)

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(merged_config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    stage = "rm" if finetuning_args.stage == "dpo" else finetuning_args.stage
    get_dataset(template, model_args, data_args, training_args, stage=stage, **tokenizer_module)
    print(f"Tokenized dataset saved at: {args.tokenized_path}")


if __name__ == "__main__":
    main()
