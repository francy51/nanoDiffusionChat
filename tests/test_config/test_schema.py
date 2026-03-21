import pytest

from src.config.schema import DatasetConfig, ExperimentConfig, ModelConfig


def test_experiment_config_validates_seq_len():
    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset=DatasetConfig(seq_len=128),
            model=ModelConfig(vocab_size=256, max_seq_len=64),
        )


def test_model_config_validates_head_divisibility():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=256, hidden_dim=130, num_layers=2, num_heads=8)
