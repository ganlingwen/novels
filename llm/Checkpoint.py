import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _replace_directory(source: Path, destination: Path) -> None:
    previous = None
    if destination.exists():
        previous = Path(tempfile.mkdtemp(prefix=f".{destination.name}-previous-", dir=destination.parent))
        previous.rmdir()
        os.replace(destination, previous)
    try:
        os.replace(source, destination)
    except Exception:
        if previous is not None:
            os.replace(previous, destination)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


@dataclass
class Checkpoint:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    gradient_accumulation: int

    global_step: int = 0
    micro_step: int = 0
    best_valid_loss: float | None = None
    best_global_step: int | None = None

    train_shuffle_generator_state: torch.Tensor | None = None
    train_shuffle_batch_offset: int = 0
    cpu_random_state: torch.Tensor | None = None
    device_random_state: torch.Tensor | None = None

    def save(self, output_dir: str | Path) -> Path:
        checkpoints = Path(output_dir) / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        path = checkpoints / f"step-{self.global_step:06d}"
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")

        temporary_path = Path(tempfile.mkdtemp(prefix=f".step-{self.global_step:06d}-", dir=checkpoints))
        try:
            self.model.save_pretrained(temporary_path, safe_serialization=True)
            self._capture_random_states()
            torch.save(self._state_dict(), temporary_path / "trainer_state.pt")
            os.replace(temporary_path, path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        return path

    def load(self, path: str | Path) -> None:
        state = torch.load(
            Path(path) / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.micro_step = state.get("micro_step", self.global_step * self.gradient_accumulation)
        self.best_valid_loss = state.get("best_valid_loss")
        self.best_global_step = state.get("best_global_step")
        if self.best_valid_loss is not None and not math.isfinite(self.best_valid_loss):
            self.best_valid_loss = None
            self.best_global_step = None
        self.train_shuffle_generator_state = state.get("train_shuffle_generator_state")
        self.train_shuffle_batch_offset = state.get("train_shuffle_batch_offset", 0)
        self.cpu_random_state = state.get("cpu_random_state")
        self.device_random_state = state.get("cuda_random_state")

    def save_best_model(self, output_dir: str | Path, valid_loss: float) -> Path | None:
        if not math.isfinite(valid_loss):
            return None
        if self.best_valid_loss is not None and valid_loss >= self.best_valid_loss:
            return None

        previous_loss = self.best_valid_loss
        previous_step = self.best_global_step
        self.best_valid_loss = valid_loss
        self.best_global_step = self.global_step
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "best"
        temporary_path = Path(tempfile.mkdtemp(prefix=".best-", dir=output_dir))
        try:
            self.model.save_pretrained(temporary_path, safe_serialization=True)
            torch.save(
                {
                    "global_step": self.best_global_step,
                    "validation_loss": self.best_valid_loss,
                },
                temporary_path / "validation_state.pt",
            )
            _replace_directory(temporary_path, path)
        except Exception:
            self.best_valid_loss = previous_loss
            self.best_global_step = previous_step
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        return path

    def training_batches(self, data_loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
        batches = self._start_training_batches(data_loader)
        while True:
            yield from self._track_batches(batches)
            generator = data_loader.generator
            if generator is None:
                raise ValueError("Training DataLoader requires a shuffle generator.")
            self.train_shuffle_generator_state = generator.get_state()
            self.train_shuffle_batch_offset = 0
            batches = iter(data_loader)

    def _start_training_batches(self, data_loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
        generator = data_loader.generator
        if generator is None:
            raise ValueError("Training DataLoader requires a shuffle generator.")
        if self.train_shuffle_generator_state is None:
            self.train_shuffle_generator_state = generator.get_state()
            return iter(data_loader)

        generator.set_state(self.train_shuffle_generator_state)
        batches = iter(data_loader)
        for _ in range(self.train_shuffle_batch_offset):
            next(batches)
        self._restore_random_states()
        return batches

    def _track_batches(self, batches: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
        for batch in batches:
            self.train_shuffle_batch_offset += 1
            yield batch

    def _state_dict(self) -> dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "best_valid_loss": self.best_valid_loss,
            "best_global_step": self.best_global_step,
            "train_shuffle_generator_state": self.train_shuffle_generator_state,
            "train_shuffle_batch_offset": self.train_shuffle_batch_offset,
            "cpu_random_state": self.cpu_random_state,
            "cuda_random_state": self.device_random_state,
        }

    def _capture_random_states(self) -> None:
        self.cpu_random_state = torch.get_rng_state()
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            self.device_random_state = torch.cuda.get_rng_state(device)

    def _restore_random_states(self) -> None:
        if self.cpu_random_state is not None:
            torch.set_rng_state(self.cpu_random_state)
        device = next(self.model.parameters()).device
        if self.device_random_state is not None and device.type == "cuda":
            torch.cuda.set_rng_state(self.device_random_state, device)
