# Quick Reference

## Commands

```bash
# Install dependencies with UV
uv sync --extra dev

# Run tests
uv run pytest

# Run a notebook
uv run marimo run notebooks/00_setup.py

# Edit a notebook
uv run marimo edit notebooks/03_train.py

# Format code
uv run ruff format src/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## Config Presets

| Config | Params | Hidden | Layers | Heads | Notes |
|--------|--------|--------|--------|-------|-------|
| tiny | ~25M | 512 | 6 | 8 | Quick experiments |
| small | ~125M | 768 | 12 | 12 | Production training |

## Key Files

| File | Purpose |
|------|---------|
| `src/config/base.py` | All configuration dataclasses |
| `src/diffusion/corrupt.py` | Token masking logic |
| `src/models/denoiser.py` | Full model definition |
| `src/training/trainer.py` | Training loop |
| `src/sampling/sampler.py` | Generation logic |

## Notebook Workflow

1. `00_setup` - First time setup
2. `01_data` - Prepare dataset (run once)
3. `02_model` - Inspect model architecture
4. `03_train` - Train model
5. `04_sample` - Generate text
6. `05_eval` - Evaluate quality

## Tensor Shapes

| Variable | Shape | Notes |
|----------|-------|-------|
| `tokens` | [batch, seq_len] | Input token IDs |
| `timesteps` | [batch] | Integer diffusion steps |
| `logits` | [batch, seq_len, vocab_size] | Model output |
| `mask` | [batch, seq_len] | Boolean, True = masked |
| `mask_prob` | [batch] | Per-sample mask probability |

## Special Token IDs (GPT-2)

| Token | ID |
|-------|-----|
| `<|endoftext|>` | 50256 |
| `[MASK]` | 50256 (reuse) |
| `[PAD]` | Custom (e.g., 50257) |
