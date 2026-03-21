from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config.io import load_experiment_config
from src.sampling.sampler import DiffusionSampler
from src.tokenization.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    sampler = DiffusionSampler.from_checkpoint(checkpoint_path)
    run_dir = checkpoint_path.parent.parent
    config = load_experiment_config(run_dir / "config.json")
    tokenizer_path = (
        Path("artifacts")
        / "datasets"
        / config.dataset.source_name
        / "prepared"
        / json.loads((run_dir / "run.json").read_text())["dataset_id"]
        / "tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    prompt_tokens = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    final = None
    for step in sampler.sample(
        prompt_tokens=prompt_tokens,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
    ):
        final = step
    if final is not None:
        print(tokenizer.decode(final.tokens[0].tolist()))


if __name__ == "__main__":
    main()
