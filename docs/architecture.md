# nanoDiffusionChat Architecture

## Overview

The repo is organized as a notebook-first research workspace:

- `src/` contains reusable, typed, mostly pure logic.
- `notebooks/` are the primary interface for humans.
- `scripts/` provide thin headless wrappers for the same flows.
- `artifacts/` is the local experiment store for datasets and runs.

## Core Layers

### Config

`src/config/` defines typed dataset, model, diffusion, training, and evaluation configs.
Named presets live in `src/config/presets.py`, and run snapshots are serialized with
`src/config/io.py`.

### Store

`src/store/` owns the canonical on-disk layout and metadata for prepared datasets
and experiment runs.

### Data

`src/data/` handles source download, tokenization, sequence chunking, and torch
dataset loading. TinyStories is the first-class public source.

### Diffusion

`src/diffusion/` centralizes timestep schedules, the forward corruption policy,
masked-token training objective, and reverse sampling strategies. There is one
authoritative corruption path used by training and evaluation.

### Models

`src/models/` defines the denoiser transformer and a small factory for config-driven
construction.

### Training and Eval

`src/training/` owns the trainer loop, metric types, and checkpoint IO.
`src/eval/` provides perplexity proxy computation, qualitative generation helpers,
and run summaries.

## Artifact Model

```text
artifacts/
  datasets/<source>/prepared/<dataset_id>/
  runs/<run_id>/
```

Each run keeps:
- `config.json`
- `run.json`
- `status.json`
- `checkpoints/`
- `metrics/`
- `samples/`
- `exports/`

## Notebook Workflow

1. `00_home.py` shows environment and artifact health.
2. `01_data_lab.py` prepares dataset artifacts.
3. `02_model_lab.py` inspects presets and forward shapes.
4. `03_train_lab.py` creates runs and trains models.
5. `04_sample_lab.py` samples from checkpoints.
6. `05_eval_lab.py` evaluates checkpoints.
7. `06_runs_dashboard.py` compares recent runs.
