import torch
import numpy as np
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer
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
        
        if model_type == "swin":
            self.model = DeepFLAIRSwin(
                img_size=patch_size,
                in_channels=1,
                out_channels=1,
                feature_size=base_channels
            )
        else:
            self.model = DeepFLAIRNet(base_channels=base_channels)
            
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight, 
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )
        
        self.inferer = SlidingWindowInferer(
            roi_size=tuple(patch_size),
            sw_batch_size=4,
            overlap=(0.5, 0.5, 0.5),
            mode="gaussian"
        )

        os.makedirs("vis", exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss, metrics = self.loss_fn(y_hat, y)
        
        bs = x.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_l1", metrics["l1"], sync_dist=True, batch_size=bs)
        self.log("train_ssim_loss", metrics["ssim_loss"], sync_dist=True, batch_size=bs)
        self.log("train_grad_loss", metrics["grad_loss"], sync_dist=True, batch_size=bs)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.inferer(x, self.model)
        loss, metrics = self.loss_fn(y_hat, y)
        
        bs = x.shape[0]
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_l1", metrics["l1"], sync_dist=True, batch_size=bs)
        self.log("val_ssim_loss", metrics["ssim_loss"], sync_dist=True, batch_size=bs)
        self.log("val_grad_loss", metrics["grad_loss"], sync_dist=True, batch_size=bs)
            
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
            axes[i, 0].imshow(img, cmap='gray'); axes[i, 0].set_title(f"Input {title}")
            axes[i, 1].imshow(lbl, cmap='gray'); axes[i, 1].set_title(f"Target {title}")
            axes[i, 2].imshow(pred, cmap='gray'); axes[i, 2].set_title(f"Pred {title} (E{self.current_epoch:03d})")
            for ax in axes[i]: ax.axis('off')
            
        plt.tight_layout()
        epoch_path = f"vis/epoch_{self.current_epoch:03d}_{stage}_3view.png"
        pred_only_path = f"vis/epoch_{self.current_epoch:03d}_pred.png"
        
        plt.savefig(epoch_path)
        plt.close(fig)
        
        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_artifact"):
                run_id = logger.run_id if hasattr(logger, "run_id") else None
                if run_id:
                    logger.experiment.log_artifact(run_id=run_id, local_path=epoch_path, artifact_path="visualizations")
                else:
                    logger.experiment.log_artifact(local_path=epoch_path, artifact_path="visualizations")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        return optimizer
