from __future__ import annotations

from pathlib import Path

from src.utils.serialization import load_json, load_jsonl


def summarize_run(run_dir: Path) -> dict[str, object]:
    status = load_json(run_dir / "status.json")
    train_metrics = load_jsonl(run_dir / "metrics" / "train.jsonl")
    eval_metrics = load_jsonl(run_dir / "metrics" / "eval.jsonl")
    latest_eval = eval_metrics[-1] if eval_metrics else {}
    return {
        "run_id": status["run_id"],
        "state": status["state"],
        "step": status["step"],
        "latest_train_loss": train_metrics[-1]["loss"] if train_metrics else None,
        "latest_eval_loss": latest_eval.get("masked_loss"),
        "masked_reconstruction_ppl": latest_eval.get(
            "masked_reconstruction_ppl",
            latest_eval.get("perplexity_proxy"),
        ),
    }
