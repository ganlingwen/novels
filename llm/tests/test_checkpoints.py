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
    trainer.global_step = 7

    checkpoint = trainer.save_checkpoint()

    assert checkpoint.name == "step-000007"
    assert (checkpoint / "model.pt").is_file()
    assert (checkpoint / "trainer_state.pt").is_file()
    state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    assert state["train_shuffle_generator_state"].dtype == torch.uint8
    assert state["train_shuffle_batch_offset"] == 0
    assert state["cpu_random_state"].dtype == torch.uint8
    with pytest.raises(FileExistsError):
        trainer.save_checkpoint()
