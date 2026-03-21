from __future__ import annotations

import argparse

from src.config.schema import DatasetConfig
from src.store.dataset_store import DatasetStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-name", default="tinystories")
    parser.add_argument("--tokenizer-name", default="char")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    store = DatasetStore()
    artifact = store.create_prepared_dataset(
        DatasetConfig(
            source_name=args.source_name,
            tokenizer_name=args.tokenizer_name,
            seq_len=args.seq_len,
        )
    )
    print(artifact)


if __name__ == "__main__":
    main()
