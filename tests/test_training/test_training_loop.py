from torch.utils.data import DataLoader

from src.data.batching import collate_token_batches
from src.data.dataset import TokenDataset
from src.models.factory import build_model_from_experiment
from src.store.dataset_store import DatasetStore
from src.store.run_store import RunStore
from src.training.loop import Trainer
from src.utils.device import get_device


def test_trainer_checkpoint_roundtrip(debug_config, tinystories_raw_text):
    del tinystories_raw_text
    dataset_artifact = DatasetStore().create_prepared_dataset(debug_config.dataset)
    train_loader = DataLoader(
        TokenDataset(dataset_artifact.train_path),
        batch_size=debug_config.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_batches,
    )
    val_loader = DataLoader(
        TokenDataset(dataset_artifact.val_path),
        batch_size=debug_config.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_batches,
    )
    run = RunStore().create_run(
        debug_config, dataset_artifact.dataset_id, preset_name="debug"
    )
    trainer = Trainer(
        config=debug_config,
        model=build_model_from_experiment(debug_config),
        train_loader=train_loader,
        val_loader=val_loader,
        device=get_device(),
        run_id=run.run_id,
        run_store=RunStore(),
    )
    batch = next(iter(train_loader))
    metrics = trainer.train_step(batch)
    assert metrics.loss >= 0
    eval_metrics = trainer.evaluate()
    checkpoint = trainer.save_checkpoint(tag="best")
    assert checkpoint.exists()
    trainer.load_checkpoint(checkpoint)
    assert trainer.step == 1
    assert eval_metrics.masked_loss >= 0
