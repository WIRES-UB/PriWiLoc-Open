"""Training-only per-AP standardization for AoA-ToF feature maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Type

import torch
from torch.utils.data import Dataset, random_split

from utils.schema import DatasetType


@dataclass(frozen=True)
class PerAPStandardizer:
    """Apply fixed scalar mean and standard deviation independently per AP.

    Args:
        mean: Per-AP feature means.
        std: Per-AP feature standard deviations.
        eps: Minimum allowed standard deviation.
    """

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def __post_init__(self) -> None:
        """Validate, normalize, and freeze the configured statistics.

        Returns:
            None.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32).detach().cpu()
        std = torch.as_tensor(self.std, dtype=torch.float32).detach().cpu()
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("Per-AP mean and std must be same-shaped rank-1 tensors.")
        if mean.numel() == 0:
            raise ValueError("At least one AP statistic is required.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Per-AP standardization statistics must be finite.")
        if self.eps <= 0:
            raise ValueError("Standardization epsilon must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std.clamp_min(float(self.eps)))

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize an `[AP, AoA, ToF]` feature tensor.

        Args:
            features: Feature tensor to normalize.

        Returns:
            A standardized tensor with the same shape.
        """
        if features.ndim != 3:
            raise ValueError(
                "PerAPStandardizer expects features with shape [AP, height, width]."
            )
        n_aps = features.shape[0]
        if n_aps > self.mean.numel():
            raise ValueError(
                f"Received {n_aps} APs but statistics exist for only "
                f"{self.mean.numel()}."
            )
        mean = self.mean[:n_aps].to(device=features.device, dtype=features.dtype)
        std = self.std[:n_aps].to(device=features.device, dtype=features.dtype)
        return (features - mean[:, None, None]) / std[:, None, None]

    def as_dict(self) -> dict[str, list[float] | float]:
        """Convert the fitted statistics to logger-friendly values.

        Returns:
            A dictionary containing means, standard deviations, and epsilon.
        """
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": float(self.eps),
        }


def fit_per_ap_standardizer(
    dataset: Dataset,
    *,
    indices: Sequence[int] | None = None,
    expected_n_aps: int | None = None,
    eps: float = 1e-6,
) -> PerAPStandardizer:
    """Fit scalar per-AP statistics from selected dataset samples.

    Args:
        dataset: Dataset containing feature tensors.
        indices: Optional sample indices included in the fit.
        expected_n_aps: Optional required AP count.
        eps: Minimum allowed standard deviation.

    Returns:
        A fitted per-AP standardizer.
    """
    selected_indices = list(range(len(dataset))) if indices is None else list(indices)
    if not selected_indices:
        raise ValueError("Cannot fit standardization on an empty training split.")

    sums: torch.Tensor | None = None
    squared_sums: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    for index in selected_indices:
        features = torch.as_tensor(dataset[index].features_2d, dtype=torch.float64)
        if features.ndim != 3:
            raise ValueError(
                f"Sample {index} features must have shape [AP, height, width]."
            )
        n_aps = int(features.shape[0])
        if expected_n_aps is not None and n_aps != int(expected_n_aps):
            raise ValueError(
                f"Sample {index} has {n_aps} APs; expected {expected_n_aps}."
            )
        if sums is None:
            sums = torch.zeros(n_aps, dtype=torch.float64)
            squared_sums = torch.zeros(n_aps, dtype=torch.float64)
            counts = torch.zeros(n_aps, dtype=torch.float64)
        elif n_aps != sums.numel():
            raise ValueError("All standardization samples must have the same AP count.")
        if not torch.isfinite(features).all():
            raise ValueError(f"Sample {index} contains NaN or infinite features.")

        flattened = features.reshape(n_aps, -1)
        sums += flattened.sum(dim=1)
        squared_sums += torch.square(flattened).sum(dim=1)
        counts += flattened.shape[1]

    assert sums is not None and squared_sums is not None and counts is not None
    mean = sums / counts
    variance = torch.clamp(squared_sums / counts - torch.square(mean), min=0.0)
    std = torch.sqrt(variance)
    return PerAPStandardizer(
        mean=mean.to(torch.float32),
        std=std.to(torch.float32),
        eps=eps,
    )


def build_training_per_ap_standardizer(
    *,
    data_paths: Any,
    dataset_class: Type[Dataset],
    train_val_split: Sequence[float] | None,
    split_seed: int,
    expected_n_aps: int,
    eps: float = 1e-6,
) -> PerAPStandardizer:
    """Load raw training data and fit statistics on the effective training set.

    Args:
        data_paths: Training dataset path or paths.
        dataset_class: Dataset implementation used to load raw samples.
        train_val_split: Optional fractions defining the seeded training subset.
        split_seed: Seed used to reproduce the training subset.
        expected_n_aps: Required number of AP feature maps.
        eps: Minimum allowed standard deviation.

    Returns:
        A fitted per-AP standardizer.
    """
    dataset = dataset_class(
        data_paths,
        transform=None,
        shuffle=False,
        type=DatasetType.train,
    )
    training_indices: Sequence[int] | None = None
    if train_val_split is not None:
        if len(train_val_split) != 2:
            raise ValueError("train_val_split must contain exactly two fractions.")
        if any(fraction <= 0 for fraction in train_val_split):
            raise ValueError("train_val_split fractions must be positive.")
        if abs(sum(train_val_split) - 1.0) > 1e-8:
            raise ValueError("train_val_split fractions must sum to 1.")
        training_subset, _ = random_split(
            dataset,
            list(train_val_split),
            generator=torch.Generator().manual_seed(int(split_seed)),
        )
        training_indices = training_subset.indices

    return fit_per_ap_standardizer(
        dataset,
        indices=training_indices,
        expected_n_aps=expected_n_aps,
        eps=eps,
    )
