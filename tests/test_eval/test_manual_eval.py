from src.eval.manual import (
    append_manual_eval_record,
    build_manual_eval_record,
    load_manual_eval_records,
    summarize_manual_eval_records,
)


def test_manual_eval_records_round_trip(tmp_path):
    run_dir = tmp_path / "artifacts" / "runs" / "run_1"
    record = build_manual_eval_record(
        run_id="run_1",
        checkpoint_path="checkpoints/last.pt",
        comparison_checkpoint_path=None,
        mode="story",
        prompt_text="Tell me a story",
        system_prompt="Be gentle",
        generation_params={"temperature": 0.8},
        candidate_index=0,
        generated_text="Once upon a time",
        rubric_scores={
            "coherence": 4,
            "fluency": 5,
            "relevance": 4,
            "instruction_following": 4,
            "story_quality": 5,
        },
        failure_flags={
            "garbage_text": False,
            "repetition_loop": False,
            "off_prompt": False,
            "cut_off": False,
            "undesirable_content": False,
        },
        evaluator_notes="Solid output",
    )

    append_manual_eval_record(run_dir, record)
    rows = load_manual_eval_records(run_dir)
    summary = summarize_manual_eval_records(run_dir)

    assert len(rows) == 1
    assert rows[0]["generated_text"] == "Once upon a time"
    assert rows[0]["pass_fail"] is True
    assert summary["num_records"] == 1
    assert summary["mean_overall_score"] == 4.4
