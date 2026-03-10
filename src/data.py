import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
import itertools
from typing import Optional, Sequence, Dict, Any, List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    SpatialPadd,
    SpatialCropd,
    RandFlipd,
    RandGaussianSmoothd,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

class PatchFisherDataset(TorchDataset):
    def __init__(self, base_dataset, samples_per_subject, transform=None):
        self.base_dataset = base_dataset
        self.samples = samples_per_subject
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset) * self.samples

    def __getitem__(self, index):
        subj_idx = index // self.samples
        data = self.base_dataset[subj_idx]
        if self.transform:
            data = self.transform(data)
        return data

class GridPatchDataset(TorchDataset):
    def __init__(self, base_dataset, patch_coords, patch_size, transform=None):
        self.base_dataset = base_dataset
        self.patch_coords = patch_coords
        self.patch_size = patch_size
        self.transform = transform
        self.samples_per_subject = len(patch_coords)

    def __len__(self):
        return len(self.base_dataset) * self.samples_per_subject

    def __getitem__(self, index):
        subj_idx = index // self.samples_per_subject
        coord_idx = index % self.samples_per_subject
        
        data = self.base_dataset[subj_idx]
        coord = self.patch_coords[coord_idx]
        
        # Calculate roi_end because MONAI requires either (center, size) or (start, end)
        roi_end = [c + p for c, p in zip(coord, self.patch_size)]
        
        # Extract patch
        cropper = SpatialCropd(
            keys=["image", "label"],
            roi_start=coord,
            roi_end=roi_end
        )
        data = cropper(data)
        
        if self.transform:
            data = self.transform(data)
        return data

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
        num_samples: int = 16,
        sampling_type: str = "random",
        pin_memory: bool = True,
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
        self.num_samples = num_samples
        self.sampling_type = sampling_type
        self.sampling_stride = 32 # Hardcoded to 32
        self.pin_memory = pin_memory

    def _get_subject_list(self) -> List[Dict[str, str]]:
        subjects = sorted([
            os.path.join(self.data_dir, d) 
            for d in os.listdir(self.data_dir) 
            if os.path.isdir(os.path.join(self.data_dir, d)) and not d.startswith('.')
        ])
        valid_data = []
        for sub_dir in subjects:
            epi = os.path.join(sub_dir, "EPI_acpc.nii.gz")
            flair = os.path.join(sub_dir, "FLAIR_star.nii.gz")
            if os.path.exists(epi) and os.path.exists(flair):
                valid_data.append({"image": epi, "label": flair, "subject_id": os.path.basename(sub_dir)})
        return valid_data

    def _calculate_grid_coords(self, spatial_shape, patch_size, stride):
        grid_points = []
        for i in range(3):
            points = list(range(0, spatial_shape[i] - patch_size[i] + 1, stride))
            if not points or points[-1] + patch_size[i] < spatial_shape[i]:
                points.append(spatial_shape[i] - patch_size[i])
            grid_points.append(points)
        return list(itertools.product(*grid_points))

    def setup(self, stage: Optional[str] = None):
        valid_data = self._get_subject_list()
        if not valid_data:
            raise RuntimeError(f"No valid subject pairs found in {self.data_dir}")

        train_val_files, test_files = train_test_split(
            valid_data, test_size=self.test_split, random_state=self.random_state
        )
        train_files, val_files = train_test_split(
            train_val_files, test_size=self.val_split, random_state=self.random_state
        )

        if stage == "fit" or stage is None:
            base_train = self._get_base_dataset(train_files, self.get_volume_transforms())
            if self.sampling_type == "grid":
                coords = self._calculate_grid_coords(self.padding_size, self.patch_size, self.sampling_stride)
                self.train_ds = GridPatchDataset(base_train, coords, self.patch_size, self.get_grid_patch_transforms())
            else:
                self.train_ds = PatchFisherDataset(base_train, self.num_samples, self.get_patch_transforms())
            
            self.val_ds = self._get_base_dataset(val_files, self.get_volume_transforms())
        
        if stage == "test" or stage is None:
            self.test_ds = self._get_base_dataset(test_files, self.get_volume_transforms())

    def _get_base_dataset(self, files, transforms):
        if self.cache_dir and self.cache_dir.lower() != "none":
            os.makedirs(self.cache_dir, exist_ok=True)
            return PersistentDataset(data=files, transform=transforms, cache_dir=self.cache_dir)
        elif self.cache_rate > 0:
            return CacheDataset(data=files, transform=transforms, cache_rate=self.cache_rate, num_workers=self.num_workers)
        return Dataset(data=files, transform=transforms)

    def get_volume_transforms(self):
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
            SpatialPadd(keys=["image", "label"], spatial_size=self.padding_size),
            EnsureTyped(keys=["image", "label"]),
        ])

    def get_patch_transforms(self):
        return Compose([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.patch_size,
                pos=1, neg=0,
                num_samples=1,
                image_key="image",
                image_threshold=0.03,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.3),
            EnsureTyped(keys=["image", "label"]),
        ])

    def get_grid_patch_transforms(self):
        return Compose([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.3),
            EnsureTyped(keys=["image", "label"]),
        ])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, pin_memory=self.pin_memory)
