from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.eval.metrics import compute_perplexity

__all__ = ["compute_perplexity"]
