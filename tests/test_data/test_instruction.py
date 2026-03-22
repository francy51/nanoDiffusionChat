from src.config.schema import DatasetConfig
from src.data.instruction import (
    ChatMessage,
    InstructionExample,
    serialize_instruction_example,
)
from src.data.prepare import prepare_dataset


def test_serialize_instruction_example_is_deterministic():
    example = InstructionExample(
        system="Be concise",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ],
    )

    first = serialize_instruction_example(example)
    second = serialize_instruction_example(example)

    assert first == second
    assert "System: Be concise" in first
    assert "User: Hello" in first
    assert "Assistant: Hi" in first


def test_prepare_chat_transcript_dataset(tmp_path):
    raw_dir = tmp_path / "artifacts" / "datasets" / "instruction" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "instruction.jsonl"
    raw_path.write_text(
        (
            '{"system":"Be helpful","messages":[{"role":"user","content":"Hi"},'
            '{"role":"assistant","content":"Hello"}]}\n'
        ),
        encoding="utf-8",
    )

    manifest = prepare_dataset(
        DatasetConfig(
            source_name="instruction",
            tokenizer_name="char",
            seq_len=16,
            format_name="chat_transcript",
        )
    )

    assert manifest.format_name == "chat_transcript"
    assert "instruction_seq16_char" in manifest.dataset_id
