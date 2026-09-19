import pytest
import torch
from torch import nn

from Train import DataLoaderConfig, Train, TrainConfig


class TinyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_fn(items):
        return items


class IndexDataset(TinyDataset):
    def __len__(self):
        return 8


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def save_pretrained(self, path, safe_serialization=True):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")


def test_checkpoint_paths_are_unique_and_complete(tmp_path):
    model = TinyModel()
    dataset = TinyDataset()
    trainer = Train(
        model,
        dataset,
        dataset,
        TrainConfig(
            output_dir=str(tmp_path),
            max_steps=10,
            data_loader=DataLoaderConfig(num_workers=0),
        ),
    )
    trainer.checkpoint.global_step = 7

    checkpoint = trainer.checkpoint.save(trainer.output_dir)

    assert checkpoint.name == "step-000007"
    assert (checkpoint / "model.pt").is_file()
    assert (checkpoint / "trainer_state.pt").is_file()
    state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    assert state["train_shuffle_generator_state"].dtype == torch.uint8
    assert state["train_shuffle_batch_offset"] == 0
    assert state["cpu_random_state"].dtype == torch.uint8
    with pytest.raises(FileExistsError):
        trainer.checkpoint.save(trainer.output_dir)


def test_best_model_is_replaced_only_when_validation_improves(tmp_path):
    model = TinyModel()
    dataset = TinyDataset()
    trainer = Train(
        model,
        dataset,
        dataset,
        TrainConfig(
            output_dir=str(tmp_path),
            max_steps=10,
            data_loader=DataLoaderConfig(num_workers=0),
        ),
    )

    assert trainer.checkpoint.save_best_model(trainer.output_dir, float("nan")) is None
    assert not (tmp_path / "best").exists()

    trainer.checkpoint.global_step = 2
    best = trainer.checkpoint.save_best_model(trainer.output_dir, 1.5)
    assert best == tmp_path / "best"

    trainer.checkpoint.global_step = 3
    assert trainer.checkpoint.save_best_model(trainer.output_dir, 1.6) is None
    state = torch.load(best / "validation_state.pt", weights_only=False)
    assert state == {"global_step": 2, "validation_loss": 1.5}

    trainer.checkpoint.global_step = 4
    trainer.model.weight.data.fill_(2)
    assert trainer.checkpoint.save_best_model(trainer.output_dir, 1.25) == best
    state = torch.load(best / "validation_state.pt", weights_only=False)
    model_state = torch.load(best / "model.pt", weights_only=False)
    assert state == {"global_step": 4, "validation_loss": 1.25}
    assert model_state["weight"].item() == 2


def test_checkpoint_restores_shuffle_position(tmp_path):
    config = TrainConfig(
        output_dir=str(tmp_path),
        max_steps=10,
        data_loader=DataLoaderConfig(num_workers=0),
    )
    dataset = IndexDataset()
    original = Train(TinyModel(), dataset, dataset, config)
    for _ in range(3):
        next(original.train_iter)
    original.checkpoint.global_step = 7
    checkpoint = original.checkpoint.save(original.output_dir)
    expected_next_batch = next(original.train_iter)

    resumed = Train(TinyModel(), dataset, dataset, config)
    resumed.checkpoint.load(checkpoint)

    assert resumed.checkpoint.global_step == 7
    assert next(resumed.train_iter) == expected_next_batch
