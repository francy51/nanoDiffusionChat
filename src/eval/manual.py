from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.serialization import append_jsonl, load_jsonl

MANUAL_EVAL_FILENAME = "manual_eval.jsonl"


@dataclass(frozen=True)
class ManualEvalRecord:
    run_id: str
    checkpoint_path: str
    comparison_checkpoint_path: str | None
    mode: str
    prompt_text: str
    system_prompt: str
    generation_params: dict[str, Any]
    candidate_index: int
    generated_text: str
    rubric_scores: dict[str, int]
    failure_flags: dict[str, bool]
    evaluator_notes: str
    overall_score: float
    pass_fail: bool
    timestamp: str


def manual_eval_path(run_dir: Path) -> Path:
    return run_dir / "manual_eval" / MANUAL_EVAL_FILENAME


def compute_overall_score(rubric_scores: dict[str, int]) -> float:
    active_scores = [score for score in rubric_scores.values() if score > 0]
    if not active_scores:
        return 0.0
    return round(sum(active_scores) / len(active_scores), 4)


def build_manual_eval_record(
    *,
    run_id: str,
    checkpoint_path: str,
    comparison_checkpoint_path: str | None,
    mode: str,
    prompt_text: str,
    system_prompt: str,
    generation_params: dict[str, Any],
    candidate_index: int,
    generated_text: str,
    rubric_scores: dict[str, int],
    failure_flags: dict[str, bool],
    evaluator_notes: str,
) -> ManualEvalRecord:
    overall_score = compute_overall_score(rubric_scores)
    severe_failure = any(failure_flags.values())
    return ManualEvalRecord(
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        comparison_checkpoint_path=comparison_checkpoint_path,
        mode=mode,
        prompt_text=prompt_text,
        system_prompt=system_prompt,
        generation_params=generation_params,
        candidate_index=candidate_index,
        generated_text=generated_text,
        rubric_scores=rubric_scores,
        failure_flags=failure_flags,
        evaluator_notes=evaluator_notes,
        overall_score=overall_score,
        pass_fail=overall_score >= 3.5 and not severe_failure,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def append_manual_eval_record(run_dir: Path, record: ManualEvalRecord) -> None:
    append_jsonl(manual_eval_path(run_dir), asdict(record))


def load_manual_eval_records(run_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(manual_eval_path(run_dir))


def summarize_manual_eval_records(run_dir: Path) -> dict[str, float | int]:
    rows = load_manual_eval_records(run_dir)
    if not rows:
        return {
            "num_records": 0,
            "mean_overall_score": 0.0,
            "pass_rate": 0.0,
        }
    num_records = len(rows)
    mean_overall_score = sum(float(row["overall_score"]) for row in rows) / num_records
    pass_rate = sum(bool(row["pass_fail"]) for row in rows) / num_records
    return {
        "num_records": num_records,
        "mean_overall_score": round(mean_overall_score, 4),
        "pass_rate": round(pass_rate, 4),
    }
