# Quick Reference

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run marimo edit notebooks/00_home.py
uv run marimo edit notebooks/03_train_lab.py
uv run python scripts/run_prepare_data.py --seq-len 128
```

## Presets

- `debug` for smoke tests
- `tiny` for local experiments
- `small` for heavier runs

## Key Paths

- `artifacts/datasets/`
- `artifacts/runs/`
- `src/config/schema.py`
- `src/store/run_store.py`
- `src/data/prepare.py`
- `src/training/loop.py`
- `src/sampling/sampler.py`

## Notebook Workflow

1. `00_home.py`
2. `01_data_lab.py`
3. `02_model_lab.py`
4. `03_train_lab.py`
5. `04_sample_lab.py`
6. `05_eval_lab.py`
7. `06_runs_dashboard.py`
8. `07_chat_lab.py`
