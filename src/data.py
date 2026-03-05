import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Sequence, Dict, Any, List
from sklearn.model_selection import train_test_split

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    SpatialPadd,
    RandFlipd,
    RandGaussianSmoothd,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

class DeepFLAIRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        patch_size: Sequence[int] = (64, 64, 64),
        padding_size: Sequence[int] = (320, 384, 320),
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.2,
        random_state: int = 42,
        cache_rate: float = 0.0,
        cache_dir: str = "outputs/monai_cache",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.padding_size = padding_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.cache_rate = cache_rate
        self.cache_dir = cache_dir

    def _get_subject_list(self) -> List[Dict[str, str]]:
        subjects = sorted(glob.glob(os.path.join(self.data_dir, "01_*")))
        valid_data = []
        for sub_dir in subjects:
            epi = os.path.join(sub_dir, "EPI_acpc.nii.gz")
            flair = os.path.join(sub_dir, "FLAIR_star.nii.gz")
            if os.path.exists(epi) and os.path.exists(flair):
                valid_data.append({"image": epi, "label": flair, "subject_id": os.path.basename(sub_dir)})
        return valid_data

    def setup(self, stage: Optional[str] = None):
        valid_data = self._get_subject_list()
        if not valid_data:
            raise RuntimeError(f"No valid subject pairs found in {self.data_dir}")

        n_test = max(1, int(len(valid_data) * self.test_split)) if len(valid_data) > 1 else 0
        if n_test > 0 and len(valid_data) > n_test:
            train_val_files, test_files = train_test_split(
                valid_data, test_size=n_test, random_state=self.random_state
            )
        else:
            train_val_files, test_files = valid_data, []

        n_val = max(1, int(len(train_val_files) * self.val_split)) if len(train_val_files) > 1 else 0
        if n_val > 0 and len(train_val_files) > n_val:
            train_files, val_files = train_test_split(
                train_val_files, test_size=n_val, random_state=self.random_state
            )
        else:
            train_files, val_files = train_val_files, train_val_files

        def get_dataset(files, transforms):
            if self.cache_dir and self.cache_dir.lower() != "none":
                os.makedirs(self.cache_dir, exist_ok=True)
                return PersistentDataset(data=files, transform=transforms, cache_dir=self.cache_dir)
            elif self.cache_rate > 0:
                return CacheDataset(data=files, transform=transforms, cache_rate=self.cache_rate, num_workers=self.num_workers)
            return Dataset(data=files, transform=transforms)

        if stage == "fit" or stage is None:
            self.train_ds = get_dataset(train_files, self.get_train_transforms())
            self.val_ds = get_dataset(val_files, self.get_val_transforms())
        
        if stage == "test" or stage is None:
            self.test_ds = get_dataset(test_files, self.get_test_transforms())

    def get_train_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image", "label"], spatial_size=self.padding_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.3),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.patch_size,
                pos=1,
                neg=1,
                num_samples=16,
                image_key="image",
                image_threshold=0.01,
            ),
            EnsureTyped(keys=["image", "label"]),
        ])

    def get_val_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image", "label"], spatial_size=self.padding_size),
            EnsureTyped(keys=["image", "label"]),
        ])

    def get_test_transforms(self):
        return self.get_val_transforms()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=True)
