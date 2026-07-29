"""
Data module for DLoc dataset.
Data module is used to load the dataset and create dataloaders for training, validation and testing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, List, Optional, Type, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dataset import DLocDatasetV2


class DatasetType:
    """Provide string constants for the dataset stage.

    Attributes:
        train: Training dataset stage.
        val: Validation dataset stage.
        test: Test dataset stage.
    """

    train: str = "train"
    val: str = "val"
    test: str = "test"


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    """Read a value from a mapping-style or attribute-style configuration entry.

    Args:
        entry: Configuration entry to inspect.
        key: Name of the value to read.
        default: Value returned when the key is unavailable.

    Returns:
        The configured value or `default`.
    """
    if hasattr(entry, "get"):
        return entry.get(key, default)
    return getattr(entry, key, default)


class DLocDataModule(LightningDataModule):
    """Load full or split training data and isolated evaluation datasets.

    Args:
        train_data_paths: Training dataset path or paths.
        val_data_paths: Optional separate validation dataset path or paths.
        test_data_paths: Optional test dataset path or paths.
        transform: Optional transform applied by each dataset.
        batch_size: Number of samples per batch.
        num_workers: Number of loader worker processes.
        prefetch_factor: Batches prefetched by each worker.
        sequence_length: Requested sample sequence length.
        dataset_class: Dataset implementation used for each source.
        train_val_split: Split fractions used when validation is absent.
        split_seed: Seed used for deterministic splitting.
        shuffle_train: Whether training samples are shuffled.
        eval_datasets: Optional named evaluation dataset definitions.
    """

    def __init__(
        self,
        *,
        train_data_paths: Union[List[str], str],
        val_data_paths: Optional[Union[List[str], str]] = None,
        test_data_paths: Optional[Union[List[str], str]] = None,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        sequence_length: int = 20,
        dataset_class: Type[Dataset] = DLocDatasetV2,
        train_val_split: Optional[List[float]] = None,
        split_seed: int = 42,
        shuffle_train: bool = False,
        eval_datasets: Optional[List[Any]] = None,
    ):
        """Configuration for dataset.

        Args:
            train_data_paths: Training dataset path(s).
            val_data_paths: Optional separate validation dataset path(s).
            test_data_paths: Optional test dataset path(s).
            transform: Transformation applied on each dataset.
            batch_size: Number of samples per data-loader batch.
            num_workers: Number of worker processes per data loader.
            prefetch_factor: Number of batches prefetched by each worker.
            sequence_length: Requested sample sequence length.
            dataset_class: Dataset implementation used for every source.
            train_val_split: Training/validation fractions used when no validation path is supplied.
            split_seed: Seed used for splitting.
            shuffle_train: Set True to shuffle the training data.
            eval_datasets: Optional additional datasets used for evaluation.

        Returns:
            None.
        """
        super().__init__()
        self.train_data_paths = train_data_paths
        self.val_data_paths = val_data_paths
        self.test_data_paths = test_data_paths
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.sequence_length = sequence_length
        self.dataset_class = dataset_class
        self.train_val_split = list(train_val_split) if train_val_split is not None else None
        self.split_seed = int(split_seed)
        self.shuffle_train = bool(shuffle_train)
        self.eval_datasets_config = list(eval_datasets or [])

        self.eval_dataset_names: list[str] = []
        self.eval_dataset_n_aps: list[int] = []
        self.test_datasets: list[Dataset] = []
        self.split_index_hash: Optional[str] = None
        self.train_index_hash: Optional[str] = None
        self.val_index_hash: Optional[str] = None

    def _load_dataset(
        self,
        data_paths: Optional[Union[List[str], str]],
        *,
        dataset_type: str,
    ) -> Optional[Dataset]:
        """Creates a configured dataset.

        Args:
            data_paths: Dataset path or paths to load.
            dataset_type: Lifecycle stage assigned to the dataset.

        Returns:
            A dataset instance, or `None` when no path was supplied.
        """
        if not data_paths:
            return None
        return self.dataset_class(
            data_paths,
            transform=self.transform,
            shuffle=False,
            type=dataset_type,
        )

    @staticmethod
    def _indices_hash(indices: list[int]) -> str:
        """Creates a stable SHA-256 identifier for an ordered index list.

        Args:
            indices: Dataset indices to identify.

        Returns:
            A hexadecimal SHA-256 digest.
        """
        payload = json.dumps(indices, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _record_split_hashes(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """Record reproducibility hashes for a generated train/validation split.

        Args:
            train_dataset: Generated training subset.
            val_dataset: Generated validation subset.

        Returns:
            None.
        """
        train_indices = list(getattr(train_dataset, "indices", range(len(train_dataset))))
        val_indices = list(getattr(val_dataset, "indices", range(len(val_dataset))))
        self.train_index_hash = self._indices_hash(train_indices)
        self.val_index_hash = self._indices_hash(val_indices)
        combined = f"{self.train_index_hash}:{self.val_index_hash}".encode("ascii")
        self.split_index_hash = hashlib.sha256(combined).hexdigest()
        print(
            "Dataset split hashes "
            f"(seed={self.split_seed}): train={self.train_index_hash}, "
            f"val={self.val_index_hash}, combined={self.split_index_hash}"
        )

    def _setup_train_and_val(self) -> None:
        """Use full training data with explicit validation, otherwise split training.

        Returns:
            None.
        """
        dataset = self._load_dataset(self.train_data_paths, dataset_type=DatasetType.train)
        if dataset is None:
            raise ValueError("Training dataset is not provided.")

        self.split_index_hash = None
        self.train_index_hash = None
        self.val_index_hash = None

        # An explicit validation dataset takes precedence over train_val_split.
        if self.val_data_paths:
            self.train_dataset = dataset
            self.val_dataset = self._load_dataset(
                self.val_data_paths,
                dataset_type=DatasetType.val,
            )
            return

        if self.train_val_split is None:
            raise ValueError(
                "dataset.train_val_split is required when no validation "
                "dataset path is provided."
            )
        if len(self.train_val_split) != 2:
            raise ValueError("dataset.train_val_split must contain exactly two fractions.")
        if any(fraction <= 0 for fraction in self.train_val_split):
            raise ValueError("dataset.train_val_split fractions must be positive.")
        if abs(sum(self.train_val_split) - 1.0) > 1e-8:
            raise ValueError("dataset.train_val_split fractions must sum to 1.")

        generator = torch.Generator().manual_seed(self.split_seed)
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            self.train_val_split,
            generator=generator,
        )
        self._record_split_hashes(self.train_dataset, self.val_dataset)

    def _setup_test_datasets(self) -> None:
        """Build isolated test datasets and their display metadata.

        Returns:
            None.
        """
        self.test_datasets = []
        self.eval_dataset_names = []
        self.eval_dataset_n_aps = []

        if self.eval_datasets_config:
            for index, entry in enumerate(self.eval_datasets_config):
                name = str(_entry_value(entry, "name", f"dataset_{index}"))
                path = _entry_value(entry, "path")
                n_aps = int(_entry_value(entry, "n_aps", 1))
                if n_aps <= 0:
                    raise ValueError(f"Evaluation dataset {name!r} has invalid n_aps={n_aps}.")

                if path is None:
                    if not isinstance(self.val_dataset, Subset) or self.train_val_split is None:
                        raise ValueError(
                            f"Evaluation dataset {name!r} has path=null, but no "
                            "seeded held-out training split is available."
                        )
                    dataset = self.val_dataset
                else:
                    dataset = self._load_dataset(path, dataset_type=DatasetType.test)
                    if dataset is None:
                        raise ValueError(f"Evaluation dataset {name!r} could not be loaded.")

                self.test_datasets.append(dataset)
                self.eval_dataset_names.append(name)
                self.eval_dataset_n_aps.append(n_aps)
        else:
            dataset = self._load_dataset(
                self.test_data_paths,
                dataset_type=DatasetType.test,
            )
            if dataset is None:
                dataset = self.val_dataset
            self.test_datasets = [dataset]
            self.eval_dataset_names = ["test"]
            self.eval_dataset_n_aps = [1]

        # Retain the old attribute for callers that inspect it directly.
        self.test_dataset = self.test_datasets[0]

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare training, validation, and test datasets for Lightning.

        Args:
            stage: Optional Lightning lifecycle stage; all datasets are prepared
                regardless of this value.

        Returns:
            None.
        """
        self._setup_train_and_val()
        self._setup_test_datasets()

    def _create_dataloader(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
        persistent_workers: bool = False,
    ) -> DataLoader:
        """Create a data loader with worker-safe prefetch settings.

        Args:
            dataset: Dataset served by the loader.
            shuffle: Whether to randomize sample order.
            persistent_workers: Whether workers persist between epochs.

        Returns:
            A configured PyTorch data loader.
        """
        if dataset is None:
            raise ValueError("Dataset is not provided.")

        kwargs: dict[str, Any] = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = persistent_workers

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.dataset_class.collate_fn,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.

        Returns:
            The configured training data loader.
        """
        return self._create_dataloader(
            self.train_dataset,
            shuffle=self.shuffle_train,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.

        Returns:
            The configured validation data loader.
        """
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        """Create one legacy test loader or individual multi-dataset loaders.

        Returns:
            A single test loader when no evaluation list is configured,
            otherwise one loader per configured evaluation dataset.
        """
        loaders = [
            self._create_dataloader(dataset, shuffle=False)
            for dataset in self.test_datasets
        ]
        if not self.eval_datasets_config:
            return loaders[0]
        return loaders
