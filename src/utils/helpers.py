from pathlib import Path
from typing import Any

from src.utils.serialization import append_jsonl, load_jsonl


def save_json_log(data: dict[str, Any], path: Path) -> None:
    append_jsonl(path, data)


def load_json_log(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)
