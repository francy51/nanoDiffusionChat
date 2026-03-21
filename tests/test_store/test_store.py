from src.config.presets import build_experiment_config
from src.config.schema import DatasetConfig
from src.store.dataset_store import DatasetStore
from src.store.run_store import RunStore


def test_dataset_store_creates_artifact(tinystories_raw_text):
    del tinystories_raw_text
    store = DatasetStore()
    artifact = store.create_prepared_dataset(
        DatasetConfig(source_name="tinystories", tokenizer_name="char", seq_len=32)
    )
    assert artifact.train_path.exists()
    assert artifact.val_path.exists()
    assert artifact.stats_path.exists()


def test_run_store_creates_and_lists_run():
    store = RunStore()
    config = build_experiment_config("debug")
    record = store.create_run(
        config,
        dataset_id="tinystories_seq64_char",
        preset_name="debug",
    )
    assert record.config_path.exists()
    assert store.list_runs()[0].run_id == record.run_id
