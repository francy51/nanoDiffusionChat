from __future__ import annotations

from dataclasses import asdict, dataclass

from src.config.schema import ExperimentConfig
from src.models.denoiser import Denoiser
from src.sampling.sampler import DiffusionSampler
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
    sampler = DiffusionSampler(model, config, device=device)
    samples: list[QualitativeSample] = []
    for prompt in prompts:
        prompt_tokens = sampler.model.new_tensor([tokenizer.encode(prompt)]).long()
        for temperature in config.eval.temperatures:
            final_step = None
            for step in sampler.sample(
                prompt_tokens=prompt_tokens,
                num_tokens=config.dataset.seq_len // 2,
                temperature=temperature,
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
