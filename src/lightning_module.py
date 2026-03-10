import torch
import numpy as np
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer

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
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        grad_weight: float = 1.0,
        patch_size: tuple = (64, 64, 64),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture Selection
        if model_type == "swin":
            self.model = DeepFLAIRSwin(feature_size=base_channels, img_size=patch_size)
        else:
            self.model = DeepFLAIRNet(base_channels=base_channels)
            
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )
        
        self.inferer = SlidingWindowInferer(
            roi_size=tuple(patch_size),
            sw_batch_size=4,
            overlap=(0.5, 0.5, 0.5),
            mode="gaussian"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_l1", metrics["l1"], sync_dist=True, batch_size=bs)
        self.log("train_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.inferer(x, self.model)
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_l1", metrics["l1"], sync_dist=True, batch_size=bs)
        self.log("val_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        if batch_idx == 0:
            self._log_images(x, y, y_hat, "val")
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
        
        # Log to loggers
        for logger in self.loggers:
            if isinstance(logger, pl.loggers.MLFlowLogger):
                # Save temp image for MLFlow
                tmp_path = f"tmp_vis_{self.global_rank}.png"
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
