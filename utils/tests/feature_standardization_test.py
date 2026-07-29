from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, random_split

from utils.feature_standardization import (
    PerAPStandardizer,
    build_training_per_ap_standardizer,
    fit_per_ap_standardizer,
)


class FeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return SimpleNamespace(features_2d=self.samples[index])


def test_fit_and_transform_gives_each_ap_zero_mean_unit_variance():
    dataset = FeatureDataset(
        [
            torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]]),
            torch.tensor([[[5.0, 7.0]], [[6.0, 8.0]]]),
        ]
    )
    standardizer = fit_per_ap_standardizer(dataset, expected_n_aps=2)

    transformed = torch.stack(
        [standardizer(dataset[index].features_2d) for index in range(len(dataset))]
    )
    flattened = transformed.permute(1, 0, 2, 3).reshape(2, -1)

    assert standardizer.mean.tolist() == pytest.approx([4.0, 5.0])
    assert standardizer.std.tolist() == pytest.approx([5**0.5, 5**0.5])
    assert flattened.mean(dim=1).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert flattened.var(dim=1, correction=0).tolist() == pytest.approx(
        [1.0, 1.0], abs=1e-6
    )


def test_builder_fits_only_the_seeded_training_subset():
    class IndexedFeatureDataset(Dataset):
        def __init__(self, path, **kwargs):
            self.values = list(range(10))

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            value = float(self.values[index])
            return SimpleNamespace(
                features_2d=torch.tensor(
                    [[[value, value]], [[value + 100.0, value + 100.0]]]
                )
            )

    reference_dataset = IndexedFeatureDataset("train")
    training_subset, _ = random_split(
        reference_dataset,
        [0.8, 0.2],
        generator=torch.Generator().manual_seed(42),
    )
    expected = fit_per_ap_standardizer(
        reference_dataset,
        indices=training_subset.indices,
        expected_n_aps=2,
    )

    actual = build_training_per_ap_standardizer(
        data_paths="train",
        dataset_class=IndexedFeatureDataset,
        train_val_split=[0.8, 0.2],
        split_seed=42,
        expected_n_aps=2,
    )

    assert torch.equal(actual.mean, expected.mean)
    assert torch.equal(actual.std, expected.std)
    assert actual.mean[0] != pytest.approx(4.5)


def test_standardizer_rejects_more_aps_than_were_fitted():
    standardizer = PerAPStandardizer(
        mean=torch.tensor([0.0, 0.0]),
        std=torch.tensor([1.0, 1.0]),
    )

    with pytest.raises(ValueError, match="statistics exist for only 2"):
        standardizer(torch.zeros(3, 2, 2))
