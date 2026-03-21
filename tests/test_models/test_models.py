import torch

from src.models.factory import build_model_from_experiment


def test_model_forward_shape(debug_config):
    model = build_model_from_experiment(debug_config)
    tokens = torch.randint(0, debug_config.model.vocab_size, (2, 32))
    timesteps = torch.randint(0, debug_config.diffusion.num_steps, (2,))
    logits = model(tokens, timesteps)
    assert logits.shape == (2, 32, debug_config.model.vocab_size)
