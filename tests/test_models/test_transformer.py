import pytest
import torch

from src.config.base import Config
from src.models.denoiser import Denoiser
from src.models.embeddings import TimestepEmbedding, TokenEmbedding
from src.models.transformer import TransformerBlock, TransformerStack


class TestEmbeddings:
    def test_token_embedding_shape(self):
        vocab_size = 1000
        hidden_dim = 128
        batch_size = 4
        seq_len = 32

        embedding = TokenEmbedding(vocab_size, hidden_dim)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(tokens)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_timestep_embedding_shape(self):
        hidden_dim = 128
        batch_size = 4

        embedding = TimestepEmbedding(hidden_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))
        output = embedding(timesteps)

        assert output.shape == (batch_size, hidden_dim)


class TestTransformer:
    def test_transformer_block_shape(self):
        hidden_dim = 128
        num_heads = 4
        batch_size = 4
        seq_len = 32

        block = TransformerBlock(hidden_dim, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = block(x)

        assert output.shape == x.shape

    def test_transformer_stack_shape(self):
        hidden_dim = 128
        num_heads = 4
        num_layers = 6
        batch_size = 4
        seq_len = 32

        stack = TransformerStack(num_layers, hidden_dim, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = stack(x)

        assert output.shape == x.shape


class TestDenoiser:
    def test_denoiser_output_shape(self, tiny_config):
        model = Denoiser(tiny_config.model, tiny_config.diffusion.num_steps)
        batch_size = 2
        seq_len = 32

        tokens = torch.randint(0, tiny_config.model.vocab_size, (batch_size, seq_len))
        timesteps = torch.randint(0, tiny_config.diffusion.num_steps, (batch_size,))

        logits = model(tokens, timesteps)

        assert logits.shape == (batch_size, seq_len, tiny_config.model.vocab_size)

    def test_denoiser_param_count(self, tiny_config):
        model = Denoiser(tiny_config.model, tiny_config.diffusion.num_steps)
        num_params = sum(p.numel() for p in model.parameters())

        assert num_params > 0
