import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

from src.arch.unet import DeepFLAIRNet
from src.arch.swin import DeepFLAIRSwin
from src.arch.attention_unet import DeepFLAIRAttentionNet
from src.arch.unetr import DeepFLAIRUNETR
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
        activation: str = "relu",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == "swin":
            swin_features = base_channels if base_channels % 12 == 0 else 24
            self.model = DeepFLAIRSwin(feature_size=swin_features, img_size=patch_size, activation=activation)
        elif model_type == "attention_unet":
            self.model = DeepFLAIRAttentionNet(base_channels=base_channels)
        elif model_type == "unetr":
            self.model = DeepFLAIRUNETR(img_size=patch_size, feature_size=base_channels)
        else:
            self.model = DeepFLAIRNet(base_channels=base_channels)
            
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight,
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss, metrics = self.loss_fn(y_hat, y)
        
        bs = x.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_mse", metrics["mse"], batch_size=bs)
        self.log("train_ssim", metrics["ssim"], batch_size=bs)
        self.log("train_grad", metrics["grad"], batch_size=bs)
        return loss

    def _infer_window(self, x):
        return sliding_window_inference(
            inputs=x,
            roi_size=self.hparams.patch_size,
            sw_batch_size=4,
            predictor=self.model,
            overlap=0.5,
            mode="gaussian" 
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self._infer_window(x)
        loss, metrics = self.loss_fn(y_hat, y)
        
        bs = x.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=bs)
        self.log("val_mse", metrics["mse"], batch_size=bs)
        self.log("val_ssim", metrics["ssim"], batch_size=bs)
        self.log("val_grad", metrics["grad"], batch_size=bs)
        
        if batch_idx == 0:
            self._log_images(y, y_hat, "val")
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self._infer_window(x)
        loss, metrics = self.loss_fn(y_hat, y)
        
        bs = x.shape[0]
        self.log("test_loss", loss, batch_size=bs)
        self.log("test_mse", metrics["mse"], batch_size=bs)
        self.log("test_ssim", metrics["ssim"], batch_size=bs)
        self.log("test_grad", metrics["grad"], batch_size=bs)
        return loss

    def _log_images(self, y, y_hat, stage):
        lbl_vol = y[0, 0].detach().cpu().float().numpy()
        pred_vol = y_hat[0, 0].detach().cpu().float().numpy()

        # Use centroid of foreground voxels so we don't slice through padding
        nonzero = np.argwhere(lbl_vol > 0.03)
        if len(nonzero) > 0:
            c = nonzero.mean(axis=0).astype(int).tolist()
        else:
            c = [s // 2 for s in lbl_vol.shape]

        views = [
            (lbl_vol[c[0], :, :], pred_vol[c[0], :, :], "Axial"),
            (lbl_vol[:, c[1], :], pred_vol[:, c[1], :], "Sagittal"),
            (lbl_vol[:, :, c[2]], pred_vol[:, :, c[2]], "Coronal"),
        ]

        epoch = self.current_epoch
        fig, axes = plt.subplots(3, 2, figsize=(8, 12))
        for i, (lbl, pred, title) in enumerate(views):
            axes[i, 0].imshow(lbl, cmap='gray')
            axes[i, 0].set_title(f"GT {title}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(pred, cmap='gray')
            axes[i, 1].set_title(f"Pred {title} E{epoch:03d} [{pred.min():.2f},{pred.max():.2f}]")
            axes[i, 1].axis('off')
        plt.tight_layout()

        os.makedirs("vis", exist_ok=True)
        save_path = f"vis/epoch_{epoch:03d}_{stage}_3view.png"
        plt.savefig(save_path)

        loggers = self.loggers if isinstance(self.loggers, (list, tuple)) else ([self.logger] if self.logger else [])
        for logger in loggers:
            if isinstance(logger, pl.loggers.MLFlowLogger):
                logger.experiment.log_artifact(run_id=logger.run_id, local_path=save_path, artifact_path="visualizations")
            elif isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_figure(f"{stage}_3view", fig, global_step=epoch)

        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            betas=(self.hparams.beta1, self.hparams.beta2)
        )