from src.config.presets import build_experiment_config, list_presets
from src.config.schema import (
    DatasetConfig,
    DiffusionConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


class Config(ExperimentConfig):
    @classmethod
    def debug(cls) -> "Config":
        return cls(**build_experiment_config("debug").__dict__)

    @classmethod
    def tiny(cls) -> "Config":
        return cls(**build_experiment_config("tiny").__dict__)

    @classmethod
    def small(cls) -> "Config":
        return cls(**build_experiment_config("small").__dict__)


__all__ = [
    "Config",
    "DatasetConfig",
    "DiffusionConfig",
    "EvalConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "build_experiment_config",
    "list_presets",
]
