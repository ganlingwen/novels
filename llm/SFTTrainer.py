import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from Checkpoint import Checkpoint
from config import DataLoaderConfig, OptimizerConfig, TrainerConfig, build_optimizer, build_scheduler
from TrainerBase import TrainerBase


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
class SFTTrainerConfig(TrainerConfig):
    causal_right_padding: bool = False
    output_dir: str = "outputs/qwen3-4b-novel-sft"
    max_steps: int = 9000


class SFTTrainer(TrainerBase):
    def __init__(self, model, train_dataset, valid_dataset, config: SFTTrainerConfig):
        self.model = model
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.max_steps = config.max_steps
        self.gradient_accumulation = config.gradient_accumulation
        self.batch_size = config.data_loader.batch_size
        self.seed = config.seed
        self.device = next(model.parameters()).device
        super().__init__(config, output_dir=self.output_dir, device=self.device)
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
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        self.checkpoint = Checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation=self.gradient_accumulation,
            metadata={"objective": "sft", "config": asdict(config)},
        )
        self.checkpoint.train_shuffle_generator_state = self.train_shuffle_generator.get_state()
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
        return {
            "loss": loss_value,
            "lr": self.checkpoint.optimizer.param_groups[0]["lr"],
            "tokens": total_tokens,
        }

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
