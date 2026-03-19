# nanoDiffusionChat Architecture

## Overview

nanoDiffusionChat is a small-scale diffusion language model implementation using masked discrete diffusion. It follows a notebook-first architecture where Marimo notebooks orchestrate the entire workflow, while `src/` contains reusable, pure components.

## Core Architecture

### Diffusion Process

The model uses **masked discrete diffusion** where:

1. **Forward Process**: Tokens are randomly replaced with a `[MASK]` token according to a schedule
2. **Reverse Process**: A transformer model predicts the original tokens at masked positions
3. **Training**: Cross-entropy loss computed only on masked positions

```
Clean Text → [Tokenize] → [Mask Tokens] → [Denoiser Model] → [Predict Original]
                ↓              ↓                                    ↓
            Token IDs     Masked IDs + Mask                 Logits → Loss
```

### Model Architecture

```
Input Tokens ──→ Token Embedding ──┐
                                   ├──→ Sum ──→ Transformer Stack ──→ Output Head ──→ Logits
Positions ─────→ Position Embed ──┘
Timesteps ─────→ Time Embedding ──┘
```

**Key Design Choices:**
- Non-causal (bidirectional) attention - the model sees the entire sequence
- Timestep embedding added to token representations
- Standard transformer architecture with pre-layer normalization

### Masking Schedule

Three schedule options:
- **uniform**: Sample mask probability from U[0,1] per sequence
- **linear**: p(t) = t where t ∈ [0,1]
- **cosine**: p(t) = 0.5 * (1 - cos(πt))

## Component Structure

### `src/config/`
- Dataclass-based configuration
- Validates invariants at construction time
- Factory methods for preset configs (`Config.tiny()`, `Config.small()`)

### `src/diffusion/`
- `schedule.py`: Mask probability functions
- `corrupt.py`: Forward corruption (masking tokens)
- `sample.py`: Reverse sampling (iterative denoising)

### `src/models/`
- `embeddings.py`: Token, position, and timestep embeddings
- `transformer.py`: Non-causal transformer blocks
- `denoiser.py`: Full model combining embeddings + transformer + output head

### `src/training/`
- `loss.py`: Masked cross-entropy (only on masked positions)
- `trainer.py`: Training loop, checkpointing, logging

### `src/sampling/`
- `sampler.py`: Iterative denoising with temperature-based sampling

## Notebook Workflow

| Notebook | Purpose |
|----------|---------|
| `00_setup` | Verify environment, download tokenizer |
| `01_data` | Download TinyStories, tokenize, create datasets |
| `02_model` | Build model, inspect architecture |
| `03_train` | Interactive training with live loss plots |
| `04_sample` | Generate text with step-by-step visualization |
| `05_eval` | Perplexity and training log analysis |

## Data Flow

```
Raw Text (TinyStories)
    ↓
download_tinystories()
    ↓
Tokenizer.encode() → Token IDs
    ↓
Chunk into fixed-length sequences
    ↓
Save as tokenized .pt files (no raw text stored)
    ↓
DiffusionDataset → DataLoader
    ↓
Training loop
```

## Checkpointing

Checkpoints saved to `checkpoints/` with:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Current step
- Config (for reproducibility)

## Logging

Training metrics logged to `logs/training_log.jsonl`:
- One JSON object per line
- Includes: step, loss, learning rate
- Can be loaded and visualized in notebooks
