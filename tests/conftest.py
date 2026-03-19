import pytest
import torch

from src.config.base import Config, ModelConfig


@pytest.fixture
def tiny_config():
    return Config.tiny()


@pytest.fixture
def sample_tokens():
    return torch.randint(0, 1000, (4, 32))
