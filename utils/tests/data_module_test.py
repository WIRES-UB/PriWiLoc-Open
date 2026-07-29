import pytest
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from data_module import DLocDataModule


class FakeDataset(Dataset):
    lengths = {"d1": 10, "d2": 3, "d3": 4}

    def __init__(self, path, **kwargs):
        self.path = path
        self.length = self.lengths[path]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor(index)

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch)


def _datamodule():
    return DLocDataModule(
        train_data_paths="d1",
        batch_size=2,
        num_workers=0,
        dataset_class=FakeDataset,
        train_val_split=[0.8, 0.2],
        split_seed=42,
        shuffle_train=True,
        eval_datasets=[
            {"name": "d1_heldout", "path": None, "n_aps": 4},
            {"name": "d2", "path": "d2", "n_aps": 3},
            {"name": "d3", "path": "d3", "n_aps": 5},
        ],
    )


def test_seeded_split_and_eval_loader_isolation():
    first = _datamodule()
    second = _datamodule()
    first.setup()
    second.setup()

    assert len(first.train_dataset) == 8
    assert len(first.val_dataset) == 2
    assert first.split_index_hash == second.split_index_hash
    assert first.eval_dataset_names == ["d1_heldout", "d2", "d3"]
    assert first.eval_dataset_n_aps == [4, 3, 5]
    assert [len(loader.dataset) for loader in first.test_dataloader()] == [2, 3, 4]
    assert isinstance(first.train_dataloader().sampler, RandomSampler)
    assert all(
        isinstance(loader.sampler, SequentialSampler)
        for loader in first.test_dataloader()
    )


def test_legacy_defaults_keep_single_loader_and_unshuffled_training():
    data_module = DLocDataModule(
        train_data_paths="d1",
        val_data_paths="d2",
        test_data_paths="d3",
        batch_size=2,
        num_workers=0,
        dataset_class=FakeDataset,
    )
    data_module.setup()

    assert isinstance(data_module.test_dataloader(), DataLoader)
    assert isinstance(data_module.train_dataloader().sampler, SequentialSampler)


def test_explicit_validation_path_keeps_the_complete_training_dataset():
    data_module = DLocDataModule(
        train_data_paths="d1",
        val_data_paths="d2",
        test_data_paths="d3",
        batch_size=2,
        num_workers=0,
        dataset_class=FakeDataset,
        train_val_split=[0.8, 0.2],
        split_seed=42,
    )

    data_module.setup()

    assert len(data_module.train_dataset) == 10
    assert len(data_module.val_dataset) == 3
    assert data_module.split_index_hash is None


def test_missing_validation_path_requires_a_training_split():
    data_module = DLocDataModule(
        train_data_paths="d1",
        test_data_paths="d3",
        batch_size=2,
        num_workers=0,
        dataset_class=FakeDataset,
    )

    with pytest.raises(
        ValueError,
        match="train_val_split is required",
    ):
        data_module.setup()
