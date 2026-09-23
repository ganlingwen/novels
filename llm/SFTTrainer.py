import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from Checkpoint import Checkpoint


def create_run_directory(output_dir: str, checkpoint: str | None = None) -> Path:
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve()
        if checkpoint_path.parent.name == "checkpoints":
            return checkpoint_path.parent.parent
        return checkpoint_path.parent

    root = Path(output_dir).resolve()
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}-{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def tensorboard_directory(output_dir: str) -> Path:
    run_dir = Path(output_dir).resolve()
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent / "tensorboard" / run_dir.name
    return run_dir / "tensorboard" / run_dir.name


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    fused: bool = False
    learning_rate: float = 1e-5
    weight_decay: float = 0.1


@dataclass(frozen=True)
class SFTTrainerConfig:
    causal_right_padding: bool = False
    output_dir: str = "outputs/qwen3-4b-novel-sft"
    max_steps: int = 9000
    gradient_accumulation: int = 16
    valid_steps: int = 500
    save_steps: int = 1000
    save_best: bool = True
    seed: int = 42
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


class SFTTrainer:
    def __init__(self, model, train_dataset, valid_dataset, config: SFTTrainerConfig):
        self.model = model
        self.config = config
        self.output_dir = config.output_dir
        self.max_steps = config.max_steps
        self.gradient_accumulation = config.gradient_accumulation
        self.batch_size = config.data_loader.batch_size
        self.valid_steps = config.valid_steps
        self.save_steps = config.save_steps
        self.seed = config.seed
        self.device = next(model.parameters()).device
        self.train_shuffle_generator = torch.Generator()
        self.train_shuffle_generator.manual_seed(self.seed)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.data_loader.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            generator=self.train_shuffle_generator,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.data_loader.num_workers,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            fused=config.optimizer.fused,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_steps, eta_min=config.optimizer.learning_rate * 0.1
        )
        self.checkpoint = Checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation=self.gradient_accumulation,
        )
        self.checkpoint.train_shuffle_generator_state = self.train_shuffle_generator.get_state()
        self.writer = SummaryWriter(str(tensorboard_directory(self.output_dir)))
        self.train_iter = self.checkpoint.training_batches(self.train_loader)

    def train_step(self):
        self.model.train()
        self.checkpoint.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        total_tokens = 0
        for _ in range(self.gradient_accumulation):
            batch = next(self.train_iter)
            total_tokens += int(batch["attention_mask"].sum().item())
            if self.config.causal_right_padding:
                batch.pop("attention_mask")
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.model(**batch).loss / self.gradient_accumulation
            loss.backward()
            total_loss += loss.detach()
            self.checkpoint.micro_step += 1
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            self.checkpoint.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                f"Non-finite training loss before optimizer step {self.checkpoint.global_step + 1} "
                f"(micro step {self.checkpoint.micro_step})."
            )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.checkpoint.optimizer.step()
        self.checkpoint.scheduler.step()
        self.checkpoint.global_step += 1
        return loss_value, total_tokens

    @torch.no_grad()
    def valid_step(self):
        self.model.eval()
        total = 0.0
        count = 0
        for batch_index, batch in enumerate(self.valid_loader):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            if self.config.causal_right_padding:
                batch.pop("attention_mask")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.model(**batch).loss.item()
            if not math.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at global step {self.checkpoint.global_step}, batch {batch_index}."
                )
            total += loss
            count += 1
        return total / max(count, 1)

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        progress = tqdm(
            total=self.max_steps,
            initial=self.checkpoint.global_step,
            desc="train",
            unit="step",
            dynamic_ncols=True,
        )
        while self.checkpoint.global_step < self.max_steps:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            loss, tokens = self.train_step()
            torch.cuda.synchronize(self.device)
            step_s = time.perf_counter() - started
            lr = self.checkpoint.optimizer.param_groups[0]["lr"]
            tokens_per_second = tokens / max(step_s, 1e-9)
            peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            self.writer.add_scalar("train/loss", loss, self.checkpoint.global_step)
            self.writer.add_scalar("train/lr", lr, self.checkpoint.global_step)
            self.writer.add_scalar("train/step_seconds", step_s, self.checkpoint.global_step)
            self.writer.add_scalar("train/tokens_per_second", tokens_per_second, self.checkpoint.global_step)
            self.writer.add_scalar("train/peak_memory_gb", peak_memory_gb, self.checkpoint.global_step)
            self.writer.add_scalar("train/batch_size", self.batch_size, self.checkpoint.global_step)
            self.writer.add_scalar(
                "train/gradient_accumulation", self.gradient_accumulation, self.checkpoint.global_step
            )
            self.writer.add_scalar(
                "train/effective_batch_size",
                self.batch_size * self.gradient_accumulation,
                self.checkpoint.global_step,
            )
            progress.update(1)
            progress.set_postfix(
                loss=f"{loss:.4f}",
                tok_s=f"{tokens_per_second / 1000:.1f}k",
                sec=f"{step_s:.2f}",
                mem=f"{peak_memory_gb:.1f}G",
                batch=f"{self.batch_size}x{self.gradient_accumulation}",
                lr=f"{lr:.2e}",
            )
            if self.valid_steps and self.checkpoint.global_step % self.valid_steps == 0:
                val_loss = self.valid_step()
                self.writer.add_scalar("valid/loss", val_loss, self.checkpoint.global_step)
                if self.config.save_best:
                    self.checkpoint.save_best_model(self.output_dir, val_loss)
                progress.set_postfix(loss=f"{loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")
            if self.save_steps and self.checkpoint.global_step % self.save_steps == 0:
                self.checkpoint.save(self.output_dir)
        progress.close()
        self.writer.close()
