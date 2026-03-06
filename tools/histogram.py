import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.transforms import ScaleIntensityRangePercentiles, HistogramNormalize

def get_slice(data):
    """Get central axial slice."""
    return data[data.shape[0]//2, :, :]

def process_subject(subject_id, data_dir, out_dir):
    epi_path = os.path.join(data_dir, subject_id, "EPI_acpc.nii.gz")
    flair_path = os.path.join(data_dir, subject_id, "FLAIR_star.nii.gz")
    
    if not os.path.exists(epi_path) or not os.path.exists(flair_path):
        print(f"Files not found for {subject_id}")
        return

    # Load data
    epi_raw = nib.load(epi_path).get_fdata()
    flair_raw = nib.load(flair_path).get_fdata()
    
    # 1. Percentile Normalization (0.5% to 99.5%)
    p_scaler = ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
    epi_p = p_scaler(epi_raw)
    flair_p = p_scaler(flair_raw)
    
    # 2. Histogram Normalization (Equalization style)
    h_norm = HistogramNormalize(num_bins=256, min=0, max=1)
    epi_h = h_norm(epi_p) # Applying to P-normed base for stability
    flair_h = h_norm(flair_p)

    # Visualization Matrix
    # Columns: Raw, Percentile, HistNorm | Rows: EPI, FLAIR*
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    
    # --- IMAGES ---
    # EPI Slices
    axes[0, 0].imshow(get_slice(epi_raw), cmap='gray'); axes[0, 0].set_title("EPI Raw")
    axes[0, 1].imshow(get_slice(epi_p), cmap='gray'); axes[0, 1].set_title("EPI Percentile Norm")
    axes[0, 2].imshow(get_slice(epi_h), cmap='gray'); axes[0, 2].set_title("EPI Hist Norm")
    
    # FLAIR* Slices
    axes[2, 0].imshow(get_slice(flair_raw), cmap='gray'); axes[2, 0].set_title("FLAIR* Raw")
    axes[2, 1].imshow(get_slice(flair_p), cmap='gray'); axes[2, 1].set_title("FLAIR* Percentile Norm")
    axes[2, 2].imshow(get_slice(flair_h), cmap='gray'); axes[2, 2].set_title("FLAIR* Hist Norm")
    
    # --- HISTOGRAMS (Non-zero only) ---
    def plot_hist(ax, data, title, color):
        ax.hist(data[data > 0].flatten(), bins=100, color=color, alpha=0.7)
        ax.set_title(title)

    plot_hist(axes[1, 0], epi_raw, "EPI Raw Hist", "blue")
    plot_hist(axes[1, 1], epi_p, "EPI P-Norm Hist", "green")
    plot_hist(axes[1, 2], epi_h, "EPI H-Norm Hist", "red")
    
    plot_hist(axes[3, 0], flair_raw, "FLAIR* Raw Hist", "blue")
    plot_hist(axes[3, 1], flair_p, "FLAIR* P-Norm Hist", "green")
    plot_hist(axes[3, 2], flair_h, "FLAIR* H-Norm Hist", "red")

    for ax in axes.flatten():
        if "Hist" not in ax.get_title():
            ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(out_dir, f"diagnostic_report_{subject_id}.png")
    plt.savefig(output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("vis", exist_ok=True)
    process_subject("01_001", "data", "vis")
