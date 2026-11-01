"""Shared configuration and construction helpers for PyTorch training."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    fused: bool = False


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str
    max_steps: int
    gradient_accumulation: int = 16
    valid_steps: int = 500
    save_steps: int = 1000
    save_best: bool = True
    seed: int = 42
    lr_schedule: str = "cosine"
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def build_scheduler(optimizer, config: TrainerConfig):
    if config.lr_schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if config.lr_schedule == "cosine":
        lr = config.optimizer.learning_rate
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps, eta_min=lr * 0.1
        )
    raise ValueError(f"Unknown learning-rate schedule: {config.lr_schedule}")


def build_optimizer(model, config: TrainerConfig):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        fused=config.optimizer.fused,
    )
