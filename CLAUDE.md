# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepFLAIR\* is a 3D medical image synthesis pipeline that learns to convert EPI MRI scans (`EPI_acpc.nii.gz`) into FLAIR\* images (`FLAIR_star.nii.gz`) using deep learning. Built on PyTorch Lightning and MONAI.

### Purpose

This project is created for a neural networks and deep learning class, with the goal of exploring post CNN architectures (mostly RNNs and transformers). The dataset is provided by the neuroimaging lab at Cedars Sinai Medical Center. Once the training and data is generated, I need to write a report. The project is due on 3/13, so its essential that we work efficiently, use our compute properly and get some good data soon so that I can start the report.

### Report

The report must follow the NeurIPS2024 style. The methods should include architectures, data preprocessing, and other techniques used. Please don't worry about the report, I will do most of this by myself later. I mostly just want you to not delete data that might be important later, and make sure there are ways to reproduce results, generate images, and run inference tests.

## Pipeline

There is a total of 89 individuals on a remote server, which should be split into a training, validation, and test split. The original authors ran optuna to test many hyperparameters for learning rate, adam, batch size, and the composite objective function. To keep things simple, we don't. They ran their final model for 75 epochs.

### Preprocessing

The preprocessing includes normalization, padding to a uniform size of 320 x 384 x 320 voxels.

For training, patches are orientation flipped and blurred with gaussian filter (none, mild, or strong). Overlapped 3D patches of 64^3 are extracted at a stride of 32, yielding a consistent patch count per subject. I call this **deterministic grid**, and I do not use this in my architectures because it creates too many patches per individual, making the epochs too long. As such, I use **random patching**, with patch centers guaranteed to be positive. This is empirically thresholded at 0.03 (after normalization).

### Loss Functions

The loss functions used are mean-squared error, perceptual similarity (1-SSIM), and directional finite-difference gradient consistency averaged across each access. Gradient clipping (norm <= 1.0) is applied.

Additionally, I attempted to use L1 loss (to push black voxels to true black, somewhat successfully). To match the original paper, I have disabled this. In general, since I do not have the time or the original weights form the paper, we will leave all three losses weighted equally at 1.0.

### Architectures

I hope to use a CNN-Unet as a baseline. Following the work of a grad student at my lab (Inga Baburyan's DeepFLAIR\*), I intend to test other architectures to see if they could be useful. For starters, I would like to test the Swin-UNETR.

#### CNN-Unet

The standard UNET is correctly implemented. It uses a 4 layer U-net with double convolutional layers, batchnorm, and LeakyReLU. The final layer is a 1x1x1 convolution followed by a ReLu.

This final activation is of particular interest. Mathematically, using a sigmoid (or other activation) should yield better results as the final output is capped between [0, 1], which should help for SSIM loss. However, I haven't really noticed that this makes a huge difference. In particular, I have tested sigmoid and hard sigmoid. I have also considered using a clamp, but there are concerns about gradients vanishing.

In order to keep things simple between models, I suggest we use ReLU for all architectures as the final activation.

For running the CNN-Unet, I have been able to use batch sizes of up to 32 when running on a single GPU. For consistency with the original paper, we should use a channel size of 16.

#### Swin-UNETR

This is the primary architecture that I intend to validate, and I have been having trouble recently getting it to work. Earlier commits had a working implementation, which is interesting because somehow gemini-cli messed things up. For the most part, it should just be just the default monai implementation with a ReLU at the end.

I also tried using batchnorm. In general, because I can only run the Swin-UNETR with a batch size of 2 or 4 due to memory constraints, its probably a better idea to use instance norm. But I have tested instance norm (no activation), batch norm + sigmoid, batch norm + hard sigmoid. Batchnorm + sigmoid actually showed non-black backgrounds, even with L1 loss. Hard sigmoid worked to get the black background, but surprisingly the Instance norm with no activation looked the best, so I'm curious to see if a ReLu would help. In general, batchnorm seemed to make things to hyperintense.


#### Other vision architectures

If I have time, I would like to test some other interesting ideas including:
 - Attention gates: instead of standard skip connections, use Attention U-Net gates, to allow the decoder to ignore irrelevant anatomical features and focus on where the contrast is more critical
 - UNETR++: Swapping the CNN encoder for Voxel-Focused Attention, theoretically low parameters but still stong in performance.
 - DADC Layers: Inject Dual Attention and Depthwise Convolutions, to preserve high-frequency texture of brain tissue

In general I am not interested in changing other aspects of the pipeline, including the loss, training strategy (diffusion), preprocessing, etc...

## Tools

### Connecting to the remote server

For connecting to the remote terminal, I have setup credentials. The server is `titan2`, and the full connection is `shena2@10.17.165.61` if this is needed, and I can provide the password. I generally run it locally through a tunnel connected to port 5000 such that I can access the MLFlow data. Don't worry about this part, because I can handle it.

### Remote server compute

We have 4 NVIDIA Quadro RTX 6000 GPUs, with 24gb memory each. We have a threadripper with 32 cores 3.7 GHz, and 256 GB memory. Sometimes zombie processess can be trouble, so make sure to check if processes are running and kill them. My user is shena2.

### Tmux

On the remote server, I usually run one tmux window called mlflow which hosts the mlflow site. I have one for each architecture, eg. swin, unet, in which I run each architecture.

### Github

Please push to the github, and pull from the remote server when making changes.

---

## Next Steps (post-training)

### Running test evaluation
Once runs are stopped, evaluate each model on the held-out test set:
```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.train --model_type unet --base_channels 16 --test \
  --ckpt_path outputs/DF_unet_BS16/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=1 $PYTHON -m src.train --model_type swin --base_channels 24 --test \
  --ckpt_path outputs/DF_swin_BS2/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=2 $PYTHON -m src.train --model_type attention_unet --base_channels 16 --test \
  --ckpt_path outputs/DF_attention_unet_BS16/checkpoints/last.ckpt

CUDA_VISIBLE_DEVICES=3 $PYTHON -m src.train --model_type unetrpp --base_channels 32 --test \
  --ckpt_path outputs/DF_unetrpp_BS32/checkpoints/last.ckpt
```
Test metrics (test_mse, test_ssim, test_grad) are logged to MLflow automatically.
Use `best` checkpoint instead of `last` if val_loss was lower earlier in training.

### Inference / visualization script (TODO)
A dedicated `scripts/infer.py` does not yet exist. It should:
- Load a checkpoint and run sliding window inference on test subjects
- Save output as `.nii.gz` for inspection in ITK-SNAP or similar
- Generate a publication-quality side-by-side figure: Input EPI / GT FLAIR* / Predicted FLAIR*
- Run across all 4 models for comparison

### Optuna preset
Hyperparameters from Baburyan et al. Optuna sweep are saved as `--preset optuna`:
- lr=0.00028, beta1=0.468, beta2=0.94
- mse_weight=1.5, ssim_weight=0.2, grad_weight=0.15
Use this for any re-runs or ablation experiments.

### Current training status (as of 2026-03-12)
All 4 models still running on titan2, max_epochs=300:
- GPU 0: unet (DF_unet_BS16) — ~52 epochs, val_loss ~0.088, still improving
- GPU 1: swin (DF_swin_BS2) — ~91 epochs, val_loss ~0.076, best so far
- GPU 2: attention_unet (DF_attention_unet_BS16) — ~63 epochs, recovering from glow artifact
- GPU 3: unetrpp (DF_unetrpp_BS32) — ~31 epochs, val_loss ~0.111, blurry (VFA too local)
Checkpoints in `outputs/<experiment_name>/checkpoints/`, vis in `vis/` (server only).

### Report notes
- SSIM loss = 1-SSIM, so lower = better. Report actual SSIM = 1 - val_ssim.
- The AttnUNet glow/collapse at epoch 63 is worth mentioning as a training dynamic.
- UNETR++ blurriness is architectural: 3×3×3 local window lacks global context for synthesis.
- See ARCHITECTURE_NOTES.md for full implementation details and citations.
