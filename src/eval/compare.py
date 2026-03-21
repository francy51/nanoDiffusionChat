from __future__ import annotations

from pathlib import Path

from src.utils.serialization import load_json, load_jsonl


def summarize_run(run_dir: Path) -> dict[str, object]:
    status = load_json(run_dir / "status.json")
    train_metrics = load_jsonl(run_dir / "metrics" / "train.jsonl")
    eval_metrics = load_jsonl(run_dir / "metrics" / "eval.jsonl")
    return {
        "run_id": status["run_id"],
        "state": status["state"],
        "step": status["step"],
        "latest_train_loss": train_metrics[-1]["loss"] if train_metrics else None,
        "latest_eval_loss": eval_metrics[-1]["masked_loss"] if eval_metrics else None,
    }
