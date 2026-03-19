from torch import Tensor, nn

from src.config.base import ModelConfig
from src.models.embeddings import PositionalEmbedding, TimestepEmbedding, TokenEmbedding
from src.models.transformer import TransformerStack


class Denoiser(nn.Module):
    def __init__(self, config: ModelConfig, num_diffusion_steps: int = 256):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = PositionalEmbedding(
            config.max_seq_len, config.hidden_dim
        )
        self.timestep_embedding = TimestepEmbedding(
            config.hidden_dim, max_steps=num_diffusion_steps
        )

        self.transformer = TransformerStack(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: Tensor, timesteps: Tensor) -> Tensor:
        tok_emb = self.token_embedding(tokens)
        pos_emb = self.position_embedding(tokens)
        time_emb = self.timestep_embedding(timesteps)

        x = tok_emb + pos_emb + time_emb.unsqueeze(1)

        x = self.transformer(x)

        logits = self.output_head(x)

        return logits
