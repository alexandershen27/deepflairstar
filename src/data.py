import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Optional, Sequence, Dict, Any, List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    SpatialPadd,
    RandFlipd,
    RandGaussianSmoothd,
    EnsureTyped,
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

class DeepFLAIRManualGridDataset(TorchDataset):
    """
    Manual Grid Tiling Dataset.
    Calculates coordinates for 64^3 patches with 50% overlap.
    Filters out patches that are pure background (black air).
    """
    def __init__(self, base_dataset, patch_size=(64, 64, 64), volume_size=(320, 384, 320), transform=None):
        self.base_dataset = base_dataset
        self.patch_size = np.array(patch_size)
        self.volume_size = np.array(volume_size)
        self.transform = transform
        self.stride = self.patch_size // 2
        
        # 1. Pre-calculate the grid coordinates
        raw_coords = []
        for z in range(0, self.volume_size[0] - self.patch_size[0] + 1, self.stride[0]):
            for y in range(0, self.volume_size[1] - self.patch_size[1] + 1, self.stride[1]):
                for x in range(0, self.volume_size[2] - self.patch_size[2] + 1, self.stride[2]):
                    raw_coords.append((z, y, x))
        
        # 2. Build the index map, FILTERING out pure background patches
        self.index_map = []
        self.coords = raw_coords
        dz, dy, dx = self.patch_size
        
        print("--- MANUAL GRID: Filtering background patches (this may take a minute)... ---")
        for sub_idx in range(len(self.base_dataset)):
            # Load the volume once to check its content
            volume_data = self.base_dataset[sub_idx]
            label_data = volume_data["label"]
            
            for coord_idx, (z, y, x) in enumerate(self.coords):
                # Check if the patch has any signal (Max > 0.05 to avoid scanning low-level noise)
                patch_max = label_data[0, z:z+dz, y:y+dy, x:x+dx].max()
                if patch_max > 0.01:
                    self.index_map.append((sub_idx, coord_idx))
                
        print(f"--- FILTER COMPLETE: Kept {len(self.index_map)} patches (Discarded ~{len(self.base_dataset)*len(raw_coords) - len(self.index_map)} black tiles) ---")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sub_idx, coord_idx = self.index_map[idx]
        volume_data = self.base_dataset[sub_idx]
        
        z, y, x = self.coords[coord_idx]
        dz, dy, dx = self.patch_size
        
        patch_item = {
            "image": volume_data["image"][:, z:z+dz, y:y+dy, x:x+dx],
            "label": volume_data["label"][:, z:z+dz, y:y+dy, x:x+dx],
            "subject_id": volume_data["subject_id"]
        }
        
        if self.transform:
            patch_item = self.transform(patch_item)
            
        return patch_item

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
            # 1. Base volumes
            base_train_ds = self._get_base_dataset(train_files, self.get_volume_transforms())
            # 2. Manual Grid with Filtering
            self.train_ds = DeepFLAIRManualGridDataset(
                base_dataset=base_train_ds,
                patch_size=self.patch_size,
                volume_size=self.padding_size,
                transform=self.get_patch_transforms()
            )
            # 3. Validation (Full volume)
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
