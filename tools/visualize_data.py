import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data import DeepFLAIRDataModule

def save_slices(image_tensor, label_tensor, output_path, subject_id, patch_idx=0):
    """
    Saves central slices (axial, sagittal, coronal) of a 3D tensor pair.
    """
    img = image_tensor.squeeze().numpy()
    lbl = label_tensor.squeeze().numpy()
    
    # Get center indices
    c = [s // 2 for s in img.shape]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Image slices
    axes[0, 0].imshow(img[c[0], :, :], cmap='gray')
    axes[0, 0].set_title(f"EPI Axial")
    axes[0, 1].imshow(img[:, c[1], :], cmap='gray')
    axes[0, 1].set_title(f"EPI Sagittal")
    axes[0, 2].imshow(img[:, :, c[2]], cmap='gray')
    axes[0, 2].set_title(f"EPI Coronal")
    
    # Label slices
    axes[1, 0].imshow(lbl[c[0], :, :], cmap='gray')
    axes[1, 0].set_title(f"FLAIR* Axial")
    axes[1, 1].imshow(lbl[:, c[1], :], cmap='gray')
    axes[1, 1].set_title(f"FLAIR* Sagittal")
    axes[1, 2].imshow(lbl[:, :, c[2]], cmap='gray')
    axes[1, 2].set_title(f"FLAIR* Coronal")
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout()
    plt.suptitle(f"Subject: {subject_id} | Patch/Volume {patch_idx}")
    plt.savefig(output_path)
    plt.close()

def main():
    data_dir = "data"
    vis_dir = "vis"
    os.makedirs(vis_dir, exist_ok=True)
    
    dm = DeepFLAIRDataModule(data_dir=data_dir, batch_size=2, num_workers=0)
    dm.setup()
    
    print("Visualizing Training Patches...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    for i in range(batch["image"].shape[0]):
        out_name = os.path.join(vis_dir, f"train_patch_{i}.png")
        save_slices(batch["image"][i], batch["label"][i], out_name, batch["subject_id"][i], i)
        print(f"Saved {out_name}")

    print("Visualizing Validation Volumes...")
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    out_name = os.path.join(vis_dir, f"val_volume_0.png")
    save_slices(batch["image"][0], batch["label"][0], out_name, batch["subject_id"][0], 0)
    print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
