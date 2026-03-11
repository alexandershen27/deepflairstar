# Architecture & Implementation Notes

DeepFLAIR* — 3D EPI → FLAIR* synthesis project.
Written for the NeurIPS 2024 report.

---

## Pipeline Overview

- **Input:** EPI_acpc.nii.gz → **Output:** FLAIR_star.nii.gz
- **Preprocessing:** ScaleIntensity [0,1], CropForeground (threshold=0.03), SpatialPad to 320×384×320
- **Training:** Random 64³ patches, 4 samples/volume, foreground-guaranteed centers
- **Augmentation:** RandFlip (axes 0,1), RandGaussianSmooth (σ=0.25–1.5, p=0.3)
- **Loss:** MSE + (1−SSIM) + gradient consistency (finite differences, L1, averaged over 3 axes), all weighted 1.0
- **Optimizer:** Adam (lr=1e-4, β=(0.5, 0.999)), gradient clipping norm≤1.0, AMP fp16
- **Validation:** Sliding window inference (overlap=0.5, Gaussian blending) on full padded volumes
- **Logging:** MLflow + TensorBoard, 3-slice visualization (Axial/Sagittal/Coronal) saved per epoch

---

## Architectures

### 1. CNN U-Net (`src/arch/unet.py`)

Standard 4-level 3D U-Net following the original DeepFLAIR* paper (Baburyan et al.).

**Encoder:** 4 × DoubleConvBlock (Conv3d → BN → LeakyReLU → Conv3d → BN → LeakyReLU), downsampled with AvgPool3d(2).
**Bottleneck:** DoubleConvBlock at 16×base_channels.
**Decoder:** ConvTranspose3d(2) upsampling, skip connections via concatenation, DoubleConvBlock.
**Final:** 1×1×1 Conv3d → ReLU.
**Init:** Kaiming normal (fan_out, leaky_relu mode).
**Channels:** base=16 → [16, 32, 64, 128, 256] per paper.

**Training config:** BS=16, repeat=8 → 32 iters/epoch.

---

### 2. Swin-UNETR (`src/arch/swin.py`)

MONAI's built-in `SwinUNETR` with a ReLU appended as the final activation.

**Key settings:**
- `feature_size=24` (must be divisible by 12 for Swin's 12-head attention)
- `norm_name="instance"` — instance norm preferred over batch norm at small batch sizes (BS=2)
- `spatial_dims=3`
- Final `nn.ReLU()` appended after the MONAI output

**Why instance norm:** At BS=2 (memory limit of the RTX 6000), batch statistics are too noisy for BatchNorm. InstanceNorm normalizes per-sample per-channel, stable at any batch size.

**Training config:** BS=2 (memory limit confirmed empirically), repeat=1 → 32 iters/epoch naturally.

**Reference:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021. MONAI implementation: `monai.networks.nets.SwinUNETR`.

---

### 3. Attention U-Net (`src/arch/attention_unet.py`)

Identical backbone to the CNN U-Net (same encoder, bottleneck, decoder channel widths), but with **soft attention gates** on every skip connection.

**Attention Gate** (Oktay et al., 2018):
Before concatenating each encoder skip with the upsampled decoder feature, an attention gate computes a soft spatial mask α ∈ [0,1]³:

```
g1  = W_g(g)     # 1×1 conv on gating signal (decoder)
x1  = W_x(x)     # 1×1 conv on skip connection (encoder)
α   = σ(ψ(ReLU(g1 + x1)))   # ψ = 1×1 conv → sigmoid
out = x * α      # elementwise multiply
```

Where `g` = upsampled decoder output (global context), `x` = encoder skip (local features).
`inter_channels = x_channels // 2` at each level.

**Intuition for synthesis:** Rather than passing all encoder features to the decoder, the network learns WHERE the contrast changes matter (e.g., periventricular regions, lesions) and suppresses irrelevant anatomy.

**4 attention gates** at resolutions: 4×, 8×, 16×, 32× downsampled.
No additional memory overhead beyond the 4 tiny 1×1 conv layers — supports same BS as vanilla U-Net.

**Training config:** BS=16, repeat=8 → 32 iters/epoch.

**Reference:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL 2018. arXiv:1804.03999.

---

### 4. UNETR++ with VFA (`src/arch/unetrpp.py`)

Custom implementation from scratch based on the VFA-UNETR++ paper.

**Reference:** Shaker et al., "UNETR++ with Voxel-Focused Attention: Efficient 3D Medical Image Segmentation with Linear-Complexity Transformers," Applied Sciences 2025. https://www.mdpi.com/2076-3417/15/20/11034
See also original UNETR++: Shaker et al., arXiv:2212.04497 / IEEE TMI 2024. https://ieeexplore.ieee.org/document/10526382

#### Voxel-Focused Attention (VFA)

Standard self-attention is O(N²) in the number of tokens. VFA reduces this to **O(27N)** by restricting each voxel to attend only to its local **3×3×3 neighborhood** (27 neighbors including itself):

```
For each token position (d,h,w):
  Q = linear(x[d,h,w])                     # [head_dim]
  K = gather 3×3×3 window around (d,h,w)   # [27, head_dim]
  V = gather 3×3×3 window around (d,h,w)   # [27, head_dim]
  α = softmax(Q·Kᵀ / √head_dim)            # [27]
  out = α·V                                 # [head_dim]
```

Window gathering is implemented efficiently in pure PyTorch via `F.pad` + 27 slice operations (no custom CUDA required).

#### EPA Block (Efficient Paired Attention)

Each transformer block runs two attention branches **in parallel with shared Q, K projections** but separate V:

```
x_norm = LayerNorm(x)
Q = W_q(x_norm),  K = W_k(x_norm)          # shared

out_spatial = VFA(Q, K, V_s=W_vs(x_norm))  # local 3×3×3 spatial
out_channel = ChanAttn(Q, K, V_c=W_vc(x_norm))  # global channel

x = x + W_proj(out_spatial + out_channel)
x = x + FFN(LayerNorm(x))
```

**Channel attention** treats each channel-head group as a token with N spatial features, computing a (head_dim × head_dim) attention matrix. Captures global inter-channel dependencies to complement VFA's local spatial focus.

#### Architecture for 64³ patches

3-stage hierarchical encoder (patch_size=4 → tokens at 16³, 8³, 4³):

```
Input [B, 1, 64, 64, 64]
  → PatchEmbed (Conv3d stride=4)  → [B, 4096, 32]  s=(16,16,16)
  → Stage 1: 2× EPABlock(32)      → skip1 [B, 32, 16, 16, 16]
  → PatchMerging (stride=2)       → [B, 512,  64]  s=(8,8,8)
  → Stage 2: 2× EPABlock(64)      → skip2 [B, 64, 8, 8, 8]
  → PatchMerging (stride=2)       → [B, 64,  128]  s=(4,4,4)
  → Stage 3: 2× EPABlock(128)     [bottleneck]
  → to_spatial                    → [B, 128, 4, 4, 4]

Decoder:
  → ConvTranspose3d(2) + cat(skip2) → DoubleConv(InstanceNorm+GELU) → [B, 64, 8, 8, 8]
  → ConvTranspose3d(2) + cat(skip1) → DoubleConv(InstanceNorm+GELU) → [B, 32, 16, 16, 16]
  → ConvTranspose3d(4)              → [B, 32, 64, 64, 64]
  → Conv1×1 + ReLU                  → [B, 1, 64, 64, 64]
```

**Parameters:** ~1.2M at base_dim=32 (paper-recommended minimum).
**Training config:** BS=32 (fits easily ~2GB peak), repeat=16 → 32 iters/epoch.

---

## Bug Fixes & Pipeline Changes

### Visualization (white image bug)
**Problem:** Gemini added `threading.Thread` for `_log_images`. Matplotlib's global state is not thread-safe — `plt.switch_backend('agg')` inside a thread corrupted the render, producing all-white images.
**Fix:** Removed threading entirely. Set `matplotlib.use('Agg')` at import time. Computation is trivial (6 small 2D slices), no performance loss.

**Secondary fix:** Center slices were computed as `s // 2` (geometric center of the padded 320×384×320 volume), which lands in zero-padded regions. Now uses the centroid of foreground voxels (`> 0.03`) so slices always cut through the brain. Epoch number (`E001`, `E002`...) added to filenames; images saved to `vis/epoch_NNN_val_3view.png` and logged to MLflow.

### Iters/epoch normalization (`--repeat_dataset`)
**Problem:** With MONAI's DataLoader, each subject is one dataset item regardless of `samples_per_volume`. At BS=16 with 64 training subjects, the U-Net only got 4 iters/epoch vs Swin's 32 — an 8× gradient update disparity over 75 epochs.
**Fix:** Added `--repeat_dataset N` which repeats the training file list N times. Since patches are random (`random_center=True`), each repeat draws genuinely different crops. All models normalized to 32 iters/epoch.

### Lambda pickling fix
`CropForegroundd(select_fn=lambda x: x > 0.03)` caused a `PicklingError` with multiprocessing DataLoader. Fixed by defining `threshold_at_zero_point_zero_three` as a named module-level function.

### Multiprocessing stability
`mp.set_sharing_strategy('file_system')` in `train.py` prevents "too many open file descriptors" errors when using MONAI's CacheDataset with multiple workers. `num_workers=4` per job (4 jobs × 4 = 16 cores, stable on the 32-core Threadripper).
