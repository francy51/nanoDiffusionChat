import torch

from src.diffusion.corrupt import MaskedDiscreteCorruptionPolicy
from src.diffusion.objectives import masked_cross_entropy
from src.diffusion.schedule import get_mask_probability, sample_timesteps


def test_sample_timesteps_shape():
    timesteps = sample_timesteps(8, 32)
    assert timesteps.shape == (8,)
    assert (timesteps >= 0).all()
    assert (timesteps < 32).all()


def test_mask_probability_boundaries():
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(get_mask_probability(t, "linear"), t)
    cosine = get_mask_probability(t, "cosine")
    assert cosine[0] == 0.0
    assert cosine[-1] == 1.0


def test_corruption_policy_masks_tokens(sample_tokens):
    policy = MaskedDiscreteCorruptionPolicy(
        num_steps=16,
        schedule_name="linear",
        mask_token_id=1,
    )
    corrupted, mask = policy.corrupt(sample_tokens, torch.tensor([0, 4, 8, 15]))
    assert corrupted.shape == sample_tokens.shape
    assert mask.shape == sample_tokens.shape
    assert ((corrupted == 1) == mask).all()


def test_masked_cross_entropy_empty_mask(sample_tokens):
    logits = torch.randn(sample_tokens.shape[0], sample_tokens.shape[1], 64)
    mask = torch.zeros_like(sample_tokens, dtype=torch.bool)
    loss = masked_cross_entropy(logits, sample_tokens, mask)
    assert loss.item() == 0.0
