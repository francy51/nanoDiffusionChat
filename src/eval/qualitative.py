from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from src.config.schema import ExperimentConfig
from src.diffusion.samplers import FullRefreshSampler
from src.models.denoiser import Denoiser
from src.tokenization.tokenizer import Tokenizer


@dataclass
class QualitativeSample:
    prompt: str
    temperature: float
    text: str


def generate_qualitative_samples(
    model: Denoiser,
    config: ExperimentConfig,
    tokenizer: Tokenizer,
    prompts: list[str],
    device: str = "cpu",
) -> list[QualitativeSample]:
    sampler = FullRefreshSampler(
        mask_token_id=config.diffusion.mask_token_id,
        device=device,
    )
    samples: list[QualitativeSample] = []
    for prompt in prompts:
        prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        for temperature in config.eval.temperatures:
            final_step = None
            for step in sampler.sample(
                model,
                prompt_tokens=prompt_tokens,
                num_new_tokens=config.dataset.seq_len // 2,
                temperature=temperature,
                num_steps=config.diffusion.num_steps,
            ):
                final_step = step
            if final_step is None:
                continue
            decoded = tokenizer.decode(final_step.tokens[0].tolist())
            samples.append(
                QualitativeSample(
                    prompt=prompt,
                    temperature=temperature,
                    text=decoded,
                )
            )
    return samples


def qualitative_samples_to_rows(
    samples: list[QualitativeSample],
) -> list[dict[str, str | float]]:
    return [asdict(sample) for sample in samples]
