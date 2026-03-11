import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import nibabel as nib
from monai.inferers import sliding_window_inference

# Multi-Model Imports
from src.arch.unet import DeepFLAIRNet
from src.arch.swin import DeepFLAIRSwin
from src.losses import DeepFLAIRLoss

class DeepFLAIRLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_type: str = "unet",
        base_channels: int = 16,
        lr: float = 1e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        mse_weight: float = 1.0,
        ssim_weight: float = 1.0,
        grad_weight: float = 1.0,
        patch_size: tuple = (64, 64, 64),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture Selection
        if model_type == "swin":
            swin_features = base_channels if base_channels % 12 == 0 else 24
            self.model = DeepFLAIRSwin(feature_size=swin_features, img_size=patch_size)
        else:
            self.model = DeepFLAIRNet(base_channels=base_channels)
            
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight,
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )
        
        # Register 3D Hann Window as a buffer
        self.register_buffer("hann_window", self._generate_3d_hann(patch_size))

    def _generate_3d_hann(self, patch_size):
        w_d = torch.hann_window(patch_size[0], periodic=False)
        w_h = torch.hann_window(patch_size[1], periodic=False)
        w_w = torch.hann_window(patch_size[2], periodic=False)
        window_3d = w_d.view(-1, 1, 1) * w_h.view(1, -1, 1) * w_w.view(1, 1, -1)
        return window_3d.unsqueeze(0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        # Functional call is the only way to pass custom maps reliably in DDP
        y_hat = sliding_window_inference(
            inputs=x,
            roi_size=self.hparams.patch_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
            mode="constant",
            roi_weight_map=self.hann_window
        )
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        if batch_idx == 0:
            self._log_images(x, y, y_hat, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = sliding_window_inference(
            inputs=x,
            roi_size=self.hparams.patch_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
            mode="constant",
            roi_weight_map=self.hann_window
        )
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        self.log("test_loss", loss, sync_dist=True, batch_size=bs)
        self.log("test_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        return loss

    def _log_images(self, x, y, y_hat, stage):
        if not self.trainer.is_global_zero:
            return
        img_vol = x[0, 0].detach().cpu().numpy()
        lbl_vol = y[0, 0].detach().cpu().numpy()
        pred_vol = y_hat[0, 0].detach().cpu().numpy()
        c = [s // 2 for s in img_vol.shape]
        views = [
            (img_vol[c[0], :, :], lbl_vol[c[0], :, :], pred_vol[c[0], :, :], "Axial"),
            (img_vol[:, c[1], :], lbl_vol[:, c[1], :], pred_vol[:, c[1], :], "Sagittal"),
            (img_vol[:, :, c[2]], lbl_vol[:, :, c[2]], pred_vol[:, :, c[2]], "Coronal")
        ]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, (img, lbl, pred, title) in enumerate(views):
            axes[i, 0].imshow(img, cmap='gray'); axes[i, 0].set_title(f"In {title}")
            axes[i, 1].imshow(lbl, cmap='gray'); axes[i, 1].set_title(f"GT {title}")
            axes[i, 2].imshow(pred, cmap='gray'); axes[i, 2].set_title(f"Pred {title}")
            for ax in axes[i]: ax.axis('off')
        plt.tight_layout()
        
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.MLFlowLogger):
                tmp_path = f"tmp_vis_{self.global_rank}_{stage}.png"
                plt.savefig(tmp_path)
                logger.experiment.log_artifact(run_id=logger.run_id, local_path=tmp_path, artifact_path="visualizations")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            elif isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(f"{stage}_3view", fig, global_step=self.global_step)
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        return optimizer
