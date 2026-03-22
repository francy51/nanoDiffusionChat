import torch
from torch import nn

from src.diffusion.samplers import ConfidenceIterativeSampler


class DeterministicModel(nn.Module):
    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, tokens, timesteps):
        del timesteps
        batch_size, seq_len = tokens.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[..., 3] = 5.0
        logits[..., 4] = 2.0
        return logits


def test_confidence_iterative_sampler_reveals_subset_first():
    sampler = ConfidenceIterativeSampler(
        mask_token_id=1,
        reveal_ratio_min=0.2,
        reveal_ratio_max=0.4,
    )
    model = DeterministicModel()

    steps = list(
        sampler.sample(
            model,
            prompt_tokens=None,
            num_new_tokens=10,
            temperature=0.0,
            num_steps=6,
        )
    )

    assert steps[0].num_masked_remaining == 10
    assert 0 < steps[1].num_revealed < 10
    assert steps[-1].num_masked_remaining == 0


def test_confidence_iterative_sampler_preserves_prompt_tokens():
    sampler = ConfidenceIterativeSampler(
        mask_token_id=1,
        reveal_ratio_min=0.25,
        reveal_ratio_max=0.5,
    )
    model = DeterministicModel()
    prompt = torch.tensor([[7, 8, 9]])

    steps = list(
        sampler.sample(
            model,
            prompt_tokens=prompt,
            num_new_tokens=6,
            temperature=0.0,
            num_steps=5,
        )
    )

    final_tokens = steps[-1].tokens[0]
    assert final_tokens[:3].tolist() == [7, 8, 9]
    assert final_tokens.min().item() >= 0
    assert final_tokens.max().item() < model.vocab_size
