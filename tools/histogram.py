import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

def create_histogram(file_path, output_path):
    print(f"Loading {file_path}...")
    data = nib.load(file_path).get_fdata().flatten()
    # Filter out zeros (background) for a clearer view of brain tissue
    data_nonzero = data[data > 0]
    
    # 1. Original Histogram
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data_nonzero, bins=100, color='blue', alpha=0.7)
    plt.title(f"Raw Intensity Histogram\n(Max: {data.max():.0f})")
    plt.xlabel("Intensity")
    plt.ylabel("Voxel Count")
    
    # 2. Percentile Normalized (Simulated)
    p05, p995 = np.percentile(data_nonzero, [0.5, 99.5])
    data_norm = np.clip((data_nonzero - p05) / (p995 - p05), 0, 1)
    
    plt.subplot(1, 2, 2)
    plt.hist(data_norm, bins=100, color='green', alpha=0.7)
    plt.title(f"Percentile Normalized [0, 1]\n(Outliers > {p995:.0f} clipped)")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Voxel Count")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

if __name__ == "__main__":
    subject = "01_001"
    f_path = f"data/{subject}/FLAIR_star.nii.gz"
    out_path = "vis/intensity_histogram.png"
    
    os.makedirs("vis", exist_ok=True)
    if os.path.exists(f_path):
        create_histogram(f_path, out_path)
    else:
        print(f"File not found: {f_path}")
