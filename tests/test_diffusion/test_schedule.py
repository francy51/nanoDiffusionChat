import pytest
import torch

from src.diffusion.schedule import get_mask_probability, sample_timesteps
from src.diffusion.corrupt import corrupt_tokens


class TestSchedule:
    def test_sample_timesteps_shape(self):
        batch_size = 8
        num_steps = 256
        t = sample_timesteps(batch_size, num_steps)
        assert t.shape == (batch_size,)
        assert (t >= 0).all() and (t < num_steps).all()

    def test_get_mask_probability_linear(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        p = get_mask_probability(t, "linear")
        assert torch.allclose(p, t)

    def test_get_mask_probability_cosine(self):
        t = torch.tensor([0.0, 0.5, 1.0])
        p = get_mask_probability(t, "cosine")
        assert p[0] == 0.0
        assert p[2] == 1.0
        assert 0 < p[1] < 1

    def test_get_mask_probability_uniform(self):
        t = torch.tensor([0.5, 0.5, 0.5])
        p = get_mask_probability(t, "uniform")
        assert p.shape == t.shape
        assert (p >= 0).all() and (p <= 1).all()


class TestCorrupt:
    def test_corrupt_tokens_shape(self):
        tokens = torch.randint(0, 100, (4, 32))
        corrupted, mask = corrupt_tokens(tokens, 0.5, mask_token_id=0)
        assert corrupted.shape == tokens.shape
        assert mask.shape == tokens.shape

    def test_corrupt_tokens_preserves_unmasked(self):
        tokens = torch.randint(1, 100, (4, 32))
        corrupted, mask = corrupt_tokens(tokens, 0.0, mask_token_id=0)
        assert torch.equal(corrupted, tokens)
        assert not mask.any()

    def test_corrupt_tokens_masks_fully(self):
        tokens = torch.randint(1, 100, (4, 32))
        corrupted, mask = corrupt_tokens(tokens, 1.0, mask_token_id=0)
        assert (corrupted == 0).all()
        assert mask.all()

    def test_corrupt_tokens_uses_mask_id(self):
        tokens = torch.randint(1, 100, (4, 32))
        mask_id = 50256
        corrupted, mask = corrupt_tokens(tokens, 1.0, mask_token_id=mask_id)
        assert (corrupted == mask_id).all()
