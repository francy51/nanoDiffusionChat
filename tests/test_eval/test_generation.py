from src.config.presets import build_experiment_config
from src.eval.compare import summarize_run
from src.eval.generation import build_chat_prompt, infer_model_mode
from src.utils.serialization import append_jsonl, save_json


def test_build_chat_prompt_keeps_latest_turns():
    messages = [
        type("Message", (), {"role": "user", "content": "old question"})(),
        type("Message", (), {"role": "assistant", "content": "old answer"})(),
        type("Message", (), {"role": "user", "content": "new question"})(),
        type("Message", (), {"role": "assistant", "content": "new answer"})(),
    ]

    prompt = build_chat_prompt(messages, "Follow instructions", max_turns=2)

    assert "old question" not in prompt
    assert "old answer" not in prompt
    assert "new question" in prompt
    assert "new answer" in prompt
    assert prompt.endswith("Assistant:")


def test_infer_model_mode_uses_dataset_format():
    base_config = build_experiment_config("story_base")
    chat_config = build_experiment_config("chat_sft_small")

    assert infer_model_mode(base_config) == "story_base"
    assert infer_model_mode(chat_config) == "chat_tuned"


def test_summarize_run_reads_legacy_and_new_metric_fields(tmp_path):
    run_dir = tmp_path / "artifacts" / "runs" / "legacy_run"
    save_json(
        run_dir / "status.json",
        {"run_id": "legacy_run", "state": "done", "step": 10},
    )
    append_jsonl(run_dir / "metrics" / "train.jsonl", {"loss": 1.0})
    append_jsonl(
        run_dir / "metrics" / "eval.jsonl",
        {"masked_loss": 0.5, "perplexity_proxy": 1.7},
    )

    legacy = summarize_run(run_dir)
    assert legacy["masked_reconstruction_ppl"] == 1.7

    append_jsonl(
        run_dir / "metrics" / "eval.jsonl",
        {"masked_loss": 0.4, "masked_reconstruction_ppl": 1.5},
    )
    updated = summarize_run(run_dir)
    assert updated["masked_reconstruction_ppl"] == 1.5
