from torch import Tensor, nn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        ff_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.ff_dim = ff_dim or hidden_dim * 4

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ff_norm = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.attention_norm(x)

        attn_output, _ = self.attention(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.ff_norm(x)
        x = residual + self.ff(x)

        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)
