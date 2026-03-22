from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class InstructionExample:
    system: str
    messages: list[ChatMessage]


def instruction_example_from_dict(payload: dict[str, object]) -> InstructionExample:
    messages_payload = payload.get("messages")
    if not isinstance(messages_payload, list):
        raise ValueError("Instruction payload must include a list of messages")

    messages = [
        ChatMessage(
            role=str(message["role"]),
            content=str(message["content"]),
        )
        for message in messages_payload
        if isinstance(message, dict)
    ]
    return InstructionExample(
        system=str(payload.get("system", "")),
        messages=messages,
    )


def serialize_instruction_example(example: InstructionExample) -> str:
    """Render an instruction example into a deterministic plain-text transcript."""
    lines: list[str] = []
    system = example.system.strip()
    if system:
        lines.append(f"System: {system}")
    for message in example.messages:
        role = message.role.strip().lower()
        if role == "assistant":
            prefix = "Assistant"
        elif role == "user":
            prefix = "User"
        else:
            prefix = role.capitalize() or "Message"
        lines.append(f"{prefix}: {message.content.strip()}")
    return "\n".join(lines)
