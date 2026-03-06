import os
import torch
import nibabel as nib
import numpy as np
from src.data import DeepFLAIRDataModule
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, SpatialPadd, EnsureTyped

def main():
    data_dir = "data"
    vis_dir = "vis"
    os.makedirs(vis_dir, exist_ok=True)
    
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=1, num_workers=0)
    subjects = dm._get_subject_list()
    test_subject = subjects[0]
    
    print(f"Testing the Final Brain Mask Threshold (0.03)...")
    
    volume_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
        SpatialPadd(keys=["image", "label"], spatial_size=(320, 384, 320)),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    processed_data = volume_transforms(test_subject)
    img_data = processed_data["image"][0].numpy()
    
    # Create mask: Everything above 0.03
    mask = (img_data >= 0.03)
    masked_data = np.zeros_like(img_data)
    masked_data[mask] = img_data[mask]
    
    affine = processed_data["image"].affine if hasattr(processed_data["image"], 'affine') else np.eye(4)
    
    out_path = os.path.join(vis_dir, "brain_mask_03_test.nii.gz")
    new_img = nib.Nifti1Image(masked_data, affine)
    nib.save(new_img, out_path)
    
    print(f"Test Mask saved to {out_path}")
    print(f"Percentage of volume kept: {100 * np.sum(mask) / img_data.size:.2f}%")

if __name__ == "__main__":
    main()
