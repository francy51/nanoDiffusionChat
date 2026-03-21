# nanoDiffusionChat

Notebook-first diffusion language-model playground with a local experiment store.

## Quick Start

```bash
uv sync --extra dev
uv run marimo edit notebooks/00_home.py
```

Primary workflow notebooks:
- `notebooks/01_data_lab.py`
- `notebooks/02_model_lab.py`
- `notebooks/03_train_lab.py`
- `notebooks/04_sample_lab.py`
- `notebooks/05_eval_lab.py`
- `notebooks/06_runs_dashboard.py`

## Repo Shape

- `src/` reusable core library
- `notebooks/` Marimo UI workflows
- `scripts/` thin headless wrappers
- `artifacts/` datasets, runs, exports
- `tests/` unit and integration coverage

## Documentation

- [Architecture](docs/architecture.md)
- [Artifact Store](docs/artifact_store.md)
- [Quick Reference](docs/quick_reference.md)
- [Technology Choices](docs/technology_choices.md)
