"""
Data module for DLoc dataset.
Data module is used to load the dataset and create dataloaders for training, validation and testing.
"""
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from dataset import DLocDatasetV2
from typing import Optional, Type, Union, List, Callable

class DatasetType:
    train: str = "train"
    val: str = "val"
    test: str = "test"

class DLocDataModule(LightningDataModule):
    def __init__(self, *,
                 train_data_paths: Union[List[str], str],
                 val_data_paths: Union[List[str], str],
                 test_data_paths: Optional[Union[List[str], str]] = None,
                 transform: Optional[Callable] = None,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 prefetch_factor: int = 2,
                 sequence_length: int = 20,
                 dataset_class: Type[Dataset] = DLocDatasetV2):
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

    def setup(self, stage: Optional[str] = None):
        def load_dataset(data_paths, shuffle=False, type=DatasetType.train):
            if not data_paths:
                return None
            return self.dataset_class(data_paths, transform=self.transform, shuffle=shuffle, type=type)

        self.train_dataset = load_dataset(self.train_data_paths)

        if not self.val_data_paths and self.test_data_paths:
            dataset = load_dataset(self.test_data_paths, type=DatasetType.test)
            self.val_dataset, self.test_dataset = random_split(dataset, [.5, .5])
        elif self.val_data_paths and not self.test_data_paths:
            dataset = load_dataset(self.val_data_paths, type=DatasetType.val)
            self.val_dataset, self.test_dataset = random_split(dataset, [.5, .5])
        else:
            self.val_dataset = load_dataset(self.val_data_paths, shuffle=False, type=DatasetType.val)
            self.test_dataset = load_dataset(self.test_data_paths, shuffle=False, type=DatasetType.test)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool, persistent_workers: bool = False) -> DataLoader:
        """
        Helper method to create a DataLoader.
        Args:
            dataset: The dataset to load.
            shuffle: Whether to shuffle the data.
            persistent_workers: Whether to keep workers alive between epochs.
                              Usually only True for training.
        Returns:
            A DataLoader instance.
        """
        if dataset is None:
            raise ValueError("Dataset is not provided.")

        return DataLoader(dataset,
                         batch_size=self.batch_size,
                         shuffle=shuffle,
                         num_workers=self.num_workers,
                         prefetch_factor=self.prefetch_factor,
                         collate_fn=DLocDatasetV2.collate_fn,
                         persistent_workers=persistent_workers)

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=False, persistent_workers=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

if __name__ == "__main__":
    train_data_path = ["/media/datadisk_2/loc_data/wifi/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind_train_2"]
    val_data_path = ["/media/datadisk_2/loc_data/wifi/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind_train_3"]
    test_data_path = ["/media/datadisk_2/loc_data/wifi/DLoc_data_split/dataset_jacobs_July28/features_aoa/ind_test_2"]
    
    data_module = DLocDataModule(train_data_paths=train_data_path,
                               val_data_paths=val_data_path, 
                               test_data_paths=test_data_path)
    data_module.setup()
    
    for batch in data_module.train_dataloader():
        features_2d, aoa_label, location_label = batch
        print(f"train dataloader size: {len(data_module.train_dataloader())}")
        print(f'Features shape: {features_2d.shape}, AOA Label shape: {aoa_label.shape}, Location Label shape: {location_label.shape}')
        break
        
    for batch in data_module.val_dataloader():
        features_2d, aoa_label, location_label = batch
        print(f"val dataloader size: {len(data_module.val_dataloader())}")
        print(f'Features shape: {features_2d.shape}, AOA Label shape: {aoa_label.shape}, Location Label shape: {location_label.shape}')
        break
        
    for batch in data_module.test_dataloader():
        features_2d, aoa_label, location_label = batch
        print(f"test dataloader size: {len(data_module.test_dataloader())}")
        print(f'Features shape: {features_2d.shape}, AOA Label shape: {aoa_label.shape}, Location Label shape: {location_label.shape}')
        break