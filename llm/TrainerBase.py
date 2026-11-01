"""Shared PyTorch training orchestration for SFT and preference trainers."""

import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class TrainerBase:
    """Own the loop; subclasses only implement one train/validation step."""

    def __init__(self, config, *, output_dir: str | Path, device: torch.device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.writer = SummaryWriter(str(self.tensorboard_path()))

    def tensorboard_path(self) -> Path:
        run_dir = self.output_dir.resolve()
        if run_dir.parent.name == "runs":
            return run_dir.parent.parent / "tensorboard" / run_dir.name
        return run_dir / "tensorboard"

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        progress = tqdm(
            total=self.config.max_steps,
            initial=self.checkpoint.global_step,
            desc=self.progress_name,
            unit="step",
            dynamic_ncols=True,
        )
        try:
            while self.checkpoint.global_step < self.config.max_steps:
                started = time.perf_counter()
                metrics = self.train_step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                metrics["step_seconds"] = time.perf_counter() - started
                if "tokens" in metrics:
                    metrics["tokens_per_second"] = metrics["tokens"] / max(metrics["step_seconds"], 1e-9)
                if self.device.type == "cuda":
                    metrics["peak_memory_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                self._write_metrics(metrics)
                step = self.checkpoint.global_step
                progress.update(1)
                progress.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}")

                if self.valid_steps and step % self.valid_steps == 0:
                    valid = self.valid_step()
                    if isinstance(valid, (int, float)):
                        valid = {"loss": valid}
                    self._write_metrics(valid, namespace="valid")
                    if getattr(self.config, "save_best", False):
                        self.checkpoint.save_best_model(self.output_dir, valid["loss"])
                    progress.set_postfix(loss=f"{metrics['loss']:.4f}", val=f"{valid['loss']:.4f}")
                if self.save_steps and step % self.save_steps == 0:
                    self.checkpoint.save(self.output_dir)
        finally:
            progress.close()
            self.writer.close()

    def _write_metrics(self, metrics: dict[str, float], *, namespace: str = "train"):
        step = self.checkpoint.global_step
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{namespace}/{name}", value, step)

    @property
    def progress_name(self):
        return "train"

    @property
    def valid_steps(self):
        return getattr(self.config, "valid_steps", 0)

    @property
    def save_steps(self):
        return getattr(self.config, "save_steps", 0)

    def train_step(self) -> dict[str, float]:
        raise NotImplementedError

    def valid_step(self) -> dict[str, float]:
        raise NotImplementedError
