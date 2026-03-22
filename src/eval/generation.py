from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.config.io import load_experiment_config
from src.config.schema import ExperimentConfig
from src.sampling.sampler import DiffusionSampler
from src.store import RunStore
from src.tokenization.tokenizer import Tokenizer


@dataclass(frozen=True)
class GenerationResources:
    run_id: str
    checkpoint_path: Path
    sampler: DiffusionSampler
    tokenizer: Tokenizer
    config: ExperimentConfig


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_index: int
    text: str
    token_ids: list[int]
    trace_rows: list[dict[str, int | str]]


def infer_model_mode(config: ExperimentConfig) -> str:
    if config.dataset.format_name == "chat_transcript":
        return "chat_tuned"
    return "story_base"


def list_checkpoint_options(run_id: str) -> list[str]:
    run = RunStore().get(run_id)
    checkpoint_dir = run.run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    return sorted(
        [path.name for path in checkpoint_dir.glob("*.pt")],
        reverse=True,
    )


def load_generation_resources(
    run_id: str,
    checkpoint_name: str | None = None,
    device: str = "cpu",
) -> GenerationResources:
    run = RunStore().get(run_id)
    if checkpoint_name is None:
        checkpoint = run.latest_checkpoint or run.best_checkpoint
    else:
        checkpoint = run.run_dir / "checkpoints" / checkpoint_name
    if checkpoint is None or not checkpoint.exists():
        raise ValueError(f"Run {run_id} has no checkpoint available")
    config = load_experiment_config(run.config_path)
    tokenizer = Tokenizer.from_file(
        Path("artifacts")
        / "datasets"
        / config.dataset.source_name
        / "prepared"
        / run.dataset_id
        / "tokenizer.json"
    )
    sampler = DiffusionSampler.from_checkpoint(checkpoint, device=device)
    return GenerationResources(
        run_id=run.run_id,
        checkpoint_path=checkpoint,
        sampler=sampler,
        tokenizer=tokenizer,
        config=config,
    )


def build_story_prompt(
    messages: list[object],
    system_prompt: str,
    max_turns: int,
) -> str:
    history = [
        message for message in messages if getattr(message, "role", None) != "system"
    ]
    lines = []
    instruction = system_prompt.strip()
    if instruction:
        lines.append(f"Story direction: {instruction}")
        lines.append("")
    for message in history[-max_turns:]:
        role = str(getattr(message, "role", "user")).lower()
        content = str(getattr(message, "content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Continuation: {content}")
        else:
            lines.append(f"Prompt: {content}")
    lines.append("Continuation:")
    return "\n".join(lines)


def build_chat_prompt(
    messages: list[object],
    system_prompt: str,
    max_turns: int,
) -> str:
    lines: list[str] = []
    if system_prompt.strip():
        lines.append(f"System: {system_prompt.strip()}")
        lines.append("")
    history = [
        message for message in messages if getattr(message, "role", None) != "system"
    ]
    for message in history[-max_turns:]:
        role = str(getattr(message, "role", "user")).capitalize()
        content = str(getattr(message, "content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
            lines.append("")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_freeform_prompt(prompt_text: str) -> str:
    return prompt_text.strip()


def prepare_prompt_text(
    mode: str,
    *,
    messages: list[object] | None = None,
    prompt_text: str = "",
    system_prompt: str = "",
    max_turns: int = 6,
) -> str:
    if mode == "chat":
        return build_chat_prompt(messages or [], system_prompt, max_turns)
    if mode == "story":
        return build_story_prompt(messages or [], system_prompt, max_turns)
    return build_freeform_prompt(prompt_text)


def generate_candidates(
    resources: GenerationResources,
    *,
    prompt_text: str,
    num_new_tokens: int,
    temperature: float,
    candidate_count: int,
    seed: int,
    sampler_name: str | None = None,
) -> list[GeneratedCandidate]:
    prompt_ids = resources.tokenizer.encode(prompt_text)
    prompt_ids = prompt_ids[
        -max(0, resources.config.model.max_seq_len - num_new_tokens) :
    ]
    prompt_tensor = (
        None if not prompt_ids else torch.tensor([prompt_ids], dtype=torch.long)
    )
    prompt_len = len(prompt_ids)
    candidates: list[GeneratedCandidate] = []

    for candidate_index in range(candidate_count):
        torch.manual_seed(seed + candidate_index)
        final_tokens = None
        trace_rows: list[dict[str, int | str]] = []
        for step in resources.sampler.sample(
            prompt_tokens=prompt_tensor,
            num_tokens=num_new_tokens,
            temperature=temperature,
            sampler_name=sampler_name,
        ):
            final_tokens = step.tokens
            trace_rows.append(
                {
                    "step_index": step.step_index,
                    "timestep": step.timestep,
                    "num_masked_remaining": step.num_masked_remaining,
                    "num_revealed": step.num_revealed,
                    "preview": resources.tokenizer.decode(step.tokens[0].tolist())[
                        :200
                    ],
                }
            )
        if final_tokens is None:
            candidates.append(
                GeneratedCandidate(
                    candidate_index=candidate_index,
                    text="",
                    token_ids=[],
                    trace_rows=trace_rows,
                )
            )
            continue
        response_ids = final_tokens[0, prompt_len:].tolist()
        candidates.append(
            GeneratedCandidate(
                candidate_index=candidate_index,
                text=resources.tokenizer.decode(response_ids).strip(),
                token_ids=response_ids,
                trace_rows=trace_rows,
            )
        )
    return candidates
