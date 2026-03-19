import json
from pathlib import Path
from typing import Any

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_json_log(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def load_json_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    logs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs
