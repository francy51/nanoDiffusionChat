# nanoDiffusionChat

A small diffusion language model implementation using masked discrete diffusion.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Run the setup notebook
uv run marimo edit notebooks/00_setup.py
```

## Project Structure

- `src/` - Core library modules (config, data, diffusion, models, training, sampling)
- `notebooks/` - Marimo notebooks for the complete workflow
- `tests/` - Test suite
- `docs/` - Documentation

## Documentation

- [Architecture](docs/architecture.md) - System design and components
- [Technology Choices](docs/technology_choices.md) - Framework decisions
- [Quick Reference](docs/quick_reference.md) - Commands and tensor shapes
- [Roadmap](docs/roadmap.md) - Development phases

## License

MIT
