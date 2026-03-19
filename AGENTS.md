# AGENTS.md

Instructions for AI coding agents working on this diffusion language model codebase.

## Build/Lint/Test Commands

```bash
# Install dependencies with UV
uv sync --extra dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_diffusion/test_schedule.py

# Run a single test by name
uv run pytest tests/test_diffusion/test_schedule.py::TestSchedule::test_sample_timesteps_shape -v

# Run tests with coverage
uv run pytest --cov=src tests/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format code
uv run ruff format src/

# Check marimo notebooks
uv run marimo check notebooks/*.py

# Run a notebook
uv run marimo run notebooks/00_setup.py

# Edit a notebook
uv run marimo edit notebooks/03_train.py
```

## Project Structure

```
src/
  config/        # Configuration classes and validation
  data/          # Dataset loading, tokenization, preprocessing
  tokenization/  # Tokenizer training and utilities
  diffusion/     # Forward corruption, reverse sampling, schedules
  models/        # Transformer denoiser, embeddings, output heads
  training/      # Training loop, checkpointing, logging
  sampling/      # Iterative denoising samplers
  eval/          # Evaluation metrics and utilities
  utils/         # Shared helpers (keep minimal)
tests/
notebooks/       # Marimo notebooks for exploration
scripts/         # CLI entry points
configs/         # YAML/JSON configuration files
```

## Code Style

### Imports

```python
# Standard library first
import os
from pathlib import Path
from typing import Optional

# Third-party next
import torch
import torch.nn as nn
from torch import Tensor

# Local imports last
from src.config.base import Config
from src.diffusion.schedule import SampleableSchedule
```

### Type Hints

Use type hints everywhere. Validate tensor shapes at boundaries.

```python
def corrupt_tokens(
    tokens: Tensor,  # [batch, seq_len]
    mask_prob: float,
    mask_token_id: int,
) -> tuple[Tensor, Tensor]:  # (corrupted_tokens, mask)
    assert tokens.dim() == 2, f"Expected 2D tensor, got {tokens.dim()}D"
    assert 0 <= mask_prob <= 1, f"mask_prob must be in [0, 1], got {mask_prob}"
    ...
```

### Naming Conventions

- Functions/methods: `snake_case` - e.g., `sample_timesteps`, `corrupt_tokens`
- Classes: `PascalCase` - e.g., `DiffusionSchedule`, `MaskedDenoiser`
- Constants: `UPPER_SNAKE_CASE` - e.g., `MAX_SEQUENCE_LENGTH`
- Private methods: `_leading_underscore`
- Descriptive names that reflect modeling concepts, not implementation details

Good: `reconstruction_loss`, `denoiser_logits`, `sample_timesteps`
Bad: `thing`, `helper2`, `run_model`, `data_stuff`

### Formatting

- Use ruff for formatting
- Max line length: 88 characters
- No trailing whitespace
- Use double quotes for strings

### Error Handling

Raise precise, informative exceptions. Never swallow errors silently.

```python
# Good
if vocab_size <= 0:
    raise ValueError(f"vocab_size must be positive, got {vocab_size}")

# Bad
if vocab_size <= 0:
    return None  # Silent failure
```

## Key Invariants to Enforce

- Token IDs must be within vocabulary range
- Mask token must exist for mask-based corruption
- Timestep values must be within `[0, num_steps - 1]`
- Tensor shapes must match model expectations
- Logits and targets must align for loss computation
- Padding tokens must be handled consistently

## Diffusion-Specific Guidelines

### Keep Diffusion Logic Explicit

Separate these concepts into distinct functions/modules:
- Timestep sampling
- Forward corruption process
- Target construction
- Denoising objective
- Reverse sampling step
- Schedule definitions

### Required Tests

- Schedule behavior at boundary timesteps (0 and max)
- Corruption at timestep 0 preserves sequence (if designed that way)
- Corruption at max timestep achieves expected masking rate
- Sampler never emits out-of-range token IDs
- Padding token handling is consistent
- Checkpoint save/load round-trips correctly

## Configuration

Use structured configs, not scattered constants:

```python
@dataclass
class ModelConfig:
    vocab_size: int
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive")
```

## Documentation

Docstrings must explain:
- Purpose
- Input contract (shapes, types, constraints)
- Output contract
- Important edge cases

```python
def sample_timesteps(batch_size: int, num_steps: int) -> Tensor:
    """Sample integer diffusion timesteps for each sequence in the batch.

    Args:
        batch_size: Number of timesteps to sample.
        num_steps: Total number of diffusion steps.

    Returns:
        Tensor of shape [batch_size] with integers in [0, num_steps - 1].
    """
```

## Marimo Notebooks

- Write cells that can be re-run cleanly (no hidden state)
- Keep expensive operations isolated in named cells
- Place core logic in `src/`, use notebooks for orchestration
- Run `marimo check` before committing

## Before Submitting Changes

1. All tests pass: `python -m pytest`
2. Code is formatted: `ruff format src/`
3. No lint errors: `ruff check src/`
4. Types check (if configured): `mypy src/`
5. Notebooks are valid: `marimo check notebooks/*.py`
6. Docstrings updated for changed interfaces

## Documentation

Project documentation is located in `docs/`:

- `docs/architecture.md` - System architecture and design decisions
- `docs/technology_choices.md` - Rationale for frameworks and tools
- `docs/roadmap.md` - Development phases and future plans
- `docs/quick_reference.md` - Common commands and tensor shapes

## What "Done" Means

A task is complete only if:
- Implementation is correct
- Code is clean and follows these guidelines
- Edge cases are handled
- Tests cover important behavior
- Documentation reflects reality
- No avoidable technical debt introduced
