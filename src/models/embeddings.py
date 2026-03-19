import math

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device)
        return self.embedding(positions)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_steps: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        half_dim = hidden_dim // 2
        emb_scale = math.log(max_steps) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb_scale)
        self.register_buffer("emb", emb)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        emb = timesteps.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)
