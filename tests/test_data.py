import os
import shutil
import pytest
import torch
import numpy as np
from src.data import DeepFLAIRDataModule

@pytest.fixture
def mock_data_dir(tmp_path):
    """Creates a mock data directory with dummy NIfTI-like structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Valid subject
    sub1 = data_dir / "01_001"
    sub1.mkdir()
    (sub1 / "EPI_acpc.nii.gz").touch()
    (sub1 / "FLAIR_star.nii.gz").touch()
    
    # Missing FLAIR*
    sub2 = data_dir / "01_002"
    sub2.mkdir()
    (sub2 / "EPI_acpc.nii.gz").touch()
    
    # Missing EPI
    sub3 = data_dir / "01_003"
    sub3.mkdir()
    (sub3 / "FLAIR_star.nii.gz").touch()
    
    return str(data_dir)

def test_subject_filtering(mock_data_dir):
    """Verify only subjects with both files are included."""
    dm = DeepFLAIRDataModule(data_dir=mock_data_dir)
    subjects = dm._get_subject_list()
    assert len(subjects) == 1
    assert subjects[0]["subject_id"] == "01_001"

def test_missing_data_raises_error(tmp_path):
    """Verify DataModule raises error if no valid data found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    dm = DeepFLAIRDataModule(data_dir=str(empty_dir))
    with pytest.raises(RuntimeError, match="No valid subject pairs found"):
        dm.setup()

@pytest.mark.parametrize("patch_size", [(32, 32, 32), (64, 64, 64)])
def test_transform_output_shapes(patch_size):
    """Live test using actual data to verify patch shapes."""
    # We use actual data because mocking NIfTI loading is complex
    data_dir = "data"
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=1, patch_size=patch_size, num_workers=0)
    dm.setup(stage="fit")
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    # Check shapes (Batch, Channel, D, H, W)
    assert batch["image"].shape == (1, 1, *patch_size)
    assert batch["label"].shape == (1, 1, *patch_size)

def test_intensity_range():
    """Verify ScaleIntensityd maps data to [0, 1]."""
    data_dir = "data"
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=1, num_workers=0)
    dm.setup(stage="fit")
    
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    
    img = batch["image"]
    lbl = batch["label"]
    
    assert torch.all(img >= 0.0) and torch.all(img <= 1.0)
    assert torch.all(lbl >= 0.0) and torch.all(lbl <= 1.0)

def test_padding_size():
    """Verify validation volumes are padded to the correct size."""
    data_dir = "data"
    padding_size = (320, 384, 320)
    dm = DeepFLAIRDataModule(data_dir=data_dir, padding_size=padding_size, num_workers=0)
    dm.setup(stage="fit")
    
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    assert batch["image"].shape == (1, 1, *padding_size)
    assert batch["label"].shape == (1, 1, *padding_size)
