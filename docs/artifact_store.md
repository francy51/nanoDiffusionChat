# Artifact Store

All persistent state lives under `artifacts/`.

## Datasets

Prepared datasets live at:

```text
artifacts/datasets/<source_name>/prepared/<dataset_id>/
```

Files:
- `dataset_manifest.json`
- `train.pt`
- `val.pt`
- `stats.json`
- `tokenizer.json`

## Runs

Runs live at:

```text
artifacts/runs/<run_id>/
```

Files:
- `run.json`
- `config.json`
- `status.json`
- `checkpoints/*.pt`
- `metrics/train.jsonl`
- `metrics/eval.jsonl`
- `samples/`
- `exports/`
