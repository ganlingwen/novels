import os
import time
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 4


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    output_dir: str = "outputs/qwen3-4b-novel-sft"
    max_steps: int = 9000
    gradient_accumulation: int = 16
    valid_steps: int = 500
    save_steps: int = 1000
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
        self.device = next(model.parameters()).device
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=config.data_loader.num_workers, collate_fn=train_dataset.collate_fn, pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=config.data_loader.num_workers, collate_fn=valid_dataset.collate_fn, pin_memory=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_steps, eta_min=config.optimizer.learning_rate * 0.1)
        self.writer = SummaryWriter(os.path.join(self.output_dir, "tensorboard"))
        self.global_step = 0
        self.micro_step = 0
        self.train_iter = iter(self.train_loader)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0
        for _ in range(self.gradient_accumulation):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            total_tokens += int(batch["attention_mask"].sum().item())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.model(**batch).loss / self.gradient_accumulation
            loss.backward()
            total_loss += loss.detach().item()
            self.micro_step += 1
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return total_loss, total_tokens

    @torch.no_grad()
    def valid_step(self):
        self.model.eval()
        total = 0.0
        count = 0
        for batch in self.valid_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total += self.model(**batch).loss.item()
            count += 1
        return total / max(count, 1)

    def save_checkpoint(self):
        path = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        torch.save({"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(), "global_step": self.global_step, "micro_step": self.micro_step}, os.path.join(path, "trainer_state.pt"))

    def load_checkpoint(self, path):
        state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.micro_step = state.get("micro_step", self.global_step * self.gradient_accumulation)

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        progress = tqdm(total=self.max_steps, initial=self.global_step, desc="train", unit="step", dynamic_ncols=True)
        last_time = time.perf_counter()
        while self.global_step < self.max_steps:
            loss, tokens = self.train_step()
            now = time.perf_counter()
            step_s = now - last_time
            last_time = now
            lr = self.optimizer.param_groups[0]["lr"]
            tokens_per_second = tokens / max(step_s, 1e-9)
            peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            self.writer.add_scalar("train/loss", loss, self.global_step)
            self.writer.add_scalar("train/lr", lr, self.global_step)
            self.writer.add_scalar("train/step_seconds", step_s, self.global_step)
            self.writer.add_scalar("train/tokens_per_second", tokens_per_second, self.global_step)
            self.writer.add_scalar("train/peak_memory_gb", peak_memory_gb, self.global_step)
            self.writer.add_scalar("train/batch_size", self.batch_size, self.global_step)
            self.writer.add_scalar("train/gradient_accumulation", self.gradient_accumulation, self.global_step)
            self.writer.add_scalar("train/effective_batch_size", self.batch_size * self.gradient_accumulation, self.global_step)
            progress.update(1)
            progress.set_postfix(loss=f"{loss:.4f}", tok_s=f"{tokens_per_second / 1000:.1f}k", sec=f"{step_s:.2f}", mem=f"{peak_memory_gb:.1f}G", batch=f"{self.batch_size}x{self.gradient_accumulation}", lr=f"{lr:.2e}")
            if self.valid_steps and self.global_step % self.valid_steps == 0:
                val_loss = self.valid_step()
                self.writer.add_scalar("valid/loss", val_loss, self.global_step)
                progress.set_postfix(loss=f"{loss:.4f}", val=f"{val_loss:.4f}", lr=f"{lr:.2e}")
            if self.save_steps and self.global_step % self.save_steps == 0:
                self.save_checkpoint()
        progress.close()
        self.writer.close()
