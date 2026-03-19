import pytest
import torch

from src.training.loss import masked_cross_entropy


class TestLoss:
    def test_masked_cross_entropy_shape(self):
        batch_size = 4
        seq_len = 32
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) > 0.5

        loss = masked_cross_entropy(logits, targets, mask, reduction="none")

        assert loss.shape == (batch_size, seq_len)

    def test_masked_cross_entropy_scalar(self):
        batch_size = 4
        seq_len = 32
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) > 0.5

        loss = masked_cross_entropy(logits, targets, mask, reduction="mean")

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_masked_cross_entropy_only_masked(self):
        batch_size = 1
        seq_len = 4
        vocab_size = 10

        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[0, 0, 5] = 10.0
        logits[0, 2, 3] = 10.0

        targets = torch.tensor([[5, 0, 3, 0]])
        mask = torch.tensor([[True, False, True, False]])

        loss = masked_cross_entropy(logits, targets, mask, reduction="mean")

        assert loss.item() < 0.01

    def test_masked_cross_entropy_empty_mask(self):
        batch_size = 4
        seq_len = 32
        vocab_size = 1000

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        loss = masked_cross_entropy(logits, targets, mask, reduction="mean")

        assert loss.item() == 0.0
