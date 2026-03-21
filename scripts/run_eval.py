import argparse

from torch.utils.data import DataLoader

from src.config.io import load_experiment_config
from src.data.batching import collate_token_batches
from src.data.dataset import TokenDataset
from src.eval.perplexity import compute_perplexity_proxy
from src.models.factory import build_model_from_experiment
from src.store import DatasetStore, RunStore
from src.training.checkpoint import load_checkpoint
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run = RunStore().get(args.run_id)
    checkpoint = run.latest_checkpoint or run.best_checkpoint
    if checkpoint is None:
        raise ValueError(f"Run {args.run_id} has no checkpoint")
    config = load_experiment_config(run.config_path)
    model = build_model_from_experiment(config)
    checkpoint_data = load_checkpoint(checkpoint, get_device())
    model.load_state_dict(checkpoint_data["model_state_dict"])
    dataset_artifact = DatasetStore().get(run.dataset_id)
    loader = DataLoader(
        TokenDataset(dataset_artifact.val_path),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_batches,
    )
    batch = next(iter(loader))
    print(compute_perplexity_proxy(model, batch, config, get_device()))


if __name__ == "__main__":
    main()
