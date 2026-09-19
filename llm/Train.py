import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


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


def checkpoint_directory(output_dir: str) -> Path:
    path = Path(output_dir) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
class TrainConfig:
    causal_right_padding: bool = False
    output_dir: str = "outputs/qwen3-4b-novel-sft"
    max_steps: int = 9000
    gradient_accumulation: int = 16
    valid_steps: int = 500
    save_steps: int = 1000
    seed: int = 42
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


class Train:
    def __init__(self, model, train_dataset, valid_dataset, config: TrainConfig):
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
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            fused=config.optimizer.fused,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_steps, eta_min=config.optimizer.learning_rate * 0.1
        )
        self.writer = SummaryWriter(os.path.join(self.output_dir, "tensorboard"))
        self.global_step = 0
        self.micro_step = 0
        self.train_shuffle_generator_state = self.train_shuffle_generator.get_state()
        self.train_shuffle_batch_offset = 0
        self.train_iter = iter(self.train_loader)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=self.device)
        total_tokens = 0
        for _ in range(self.gradient_accumulation):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_shuffle_generator_state = self.train_shuffle_generator.get_state()
                self.train_shuffle_batch_offset = 0
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
            self.train_shuffle_batch_offset += 1
            total_tokens += int(batch["attention_mask"].sum().item())
            if self.config.causal_right_padding:
                batch.pop("attention_mask")
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.model(**batch).loss / self.gradient_accumulation
            loss.backward()
            total_loss += loss.detach()
            self.micro_step += 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return total_loss.item(), total_tokens

    @torch.no_grad()
    def valid_step(self):
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.valid_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            if self.config.causal_right_padding:
                batch.pop("attention_mask")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += self.model(**batch).loss.item()
            count += 1
        return total / max(count, 1)

    def save_checkpoint(self):
        checkpoints = checkpoint_directory(self.output_dir)
        path = checkpoints / f"step-{self.global_step:06d}"
        if path.exists():
            raise FileExistsError(f"Checkpoint already exists: {path}")
        temporary_path = Path(tempfile.mkdtemp(prefix=f".step-{self.global_step:06d}-", dir=checkpoints))
        try:
            self.model.save_pretrained(temporary_path, safe_serialization=True)
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                    "micro_step": self.micro_step,
                    "train_shuffle_generator_state": self.train_shuffle_generator_state,
                    "train_shuffle_batch_offset": self.train_shuffle_batch_offset,
                    "cpu_random_state": torch.get_rng_state(),
                    "cuda_random_state": (
                        torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
                    ),
                },
                temporary_path / "trainer_state.pt",
            )
            os.replace(temporary_path, path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        return path

    def load_checkpoint(self, path):
        state = torch.load(
            os.path.join(path, "trainer_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.micro_step = state.get("micro_step", self.global_step * self.gradient_accumulation)
        self.train_shuffle_generator_state = state.get("train_shuffle_generator_state")
        self.train_shuffle_batch_offset = state.get("train_shuffle_batch_offset", 0)
        if self.train_shuffle_generator_state is not None:
            self.train_shuffle_generator.set_state(self.train_shuffle_generator_state)
            self.train_iter = iter(self.train_loader)
            for _ in range(self.train_shuffle_batch_offset):
                next(self.train_iter)
        cpu_random_state = state.get("cpu_random_state")
        if cpu_random_state is not None:
            torch.set_rng_state(cpu_random_state)
        cuda_random_state = state.get("cuda_random_state")
        if cuda_random_state is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_random_state, self.device)

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        progress = tqdm(total=self.max_steps, initial=self.global_step, desc="train", unit="step", dynamic_ncols=True)
        while self.global_step < self.max_steps:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            loss, tokens = self.train_step()
            torch.cuda.synchronize(self.device)
            step_s = time.perf_counter() - started
            lr = self.optimizer.param_groups[0]["lr"]
            tokens_per_second = tokens / max(step_s, 1e-9)
            peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            self.writer.add_scalar("train/loss", loss, self.global_step)
            self.writer.add_scalar("train/lr", lr, self.global_step)
            self.writer.add_scalar("train/step_seconds", step_s, self.global_step)
            self.writer.add_scalar("train/tokens_per_second", tokens_per_second, self.global_step)
            self.writer.add_scalar("train/peak_memory_gb", peak_memory_gb, self.global_step)
            self.writer.add_scalar("train/batch_size", self.batch_size, self.global_step)
            self.writer.add_scalar("train/gradient_accumulation", self.gradient_accumulation, self.global_step)
            self.writer.add_scalar(
                "train/effective_batch_size", self.batch_size * self.gradient_accumulation, self.global_step
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
            if self.valid_steps and self.global_step % self.valid_steps == 0:
                val_loss = self.valid_step()
                self.writer.add_scalar("valid/loss", val_loss, self.global_step)
                progress.set_postfix(loss=f"{loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")
            if self.save_steps and self.global_step % self.save_steps == 0:
                self.save_checkpoint()
        progress.close()
        self.writer.close()
