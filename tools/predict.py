import argparse
import os
import torch
import nibabel as nib
import numpy as np
from src.lightning_module import DeepFLAIRLightningModule
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType

def predict(args):
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Model from Checkpoint
    print(f"Loading model from {args.ckpt_path}...")
    # Map location 'cpu' handles loading even if GPUs differ
    model = DeepFLAIRLightningModule.load_from_checkpoint(args.ckpt_path, map_location='cpu')
    model.to(device)
    model.eval()
    
    # 3. Preprocessing (Strict match to training)
    # Note: We use volume-level transforms
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType()
    ])
    
    # 4. Run Inference
    print(f"Synthesizing FLAIR* for {args.input_path}...")
    input_tensor = transforms(args.input_path).unsqueeze(0).to(device) # Add batch dim
    
    with torch.no_grad():
        output_tensor = model.inferer(input_tensor, model)
    
    # 5. Save Result
    output_data = output_tensor.squeeze().cpu().numpy()
    
    # Use the input's affine to keep orientation correct
    input_img = nib.load(args.input_path)
    new_img = nib.Nifti1Image(output_data, input_img.affine, input_img.header)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    nib.save(new_img, args.output_path)
    print(f"Synthesis complete! Saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best .ckpt file")
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw EPI_acpc.nii.gz")
    parser.add_argument("--output_path", type=str, default="vis/synthesis_result.nii.gz")
    args = parser.parse_args()
    predict(args)
