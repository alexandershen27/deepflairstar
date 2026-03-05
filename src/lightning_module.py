import torch
import numpy as np
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer
from src.arch.unet import DeepFLAIRNet
from src.losses import DeepFLAIRLoss

class DeepFLAIRLightningModule(pl.LightningModule):
    def __init__(
        self,
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
        
        self.model = DeepFLAIRNet(base_channels=base_channels)
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight, 
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )
        
        self.inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian"
        )

        os.makedirs("vis", exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss, metrics = self.loss_fn(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        for name, val in metrics.items():
            self.log(f"train_{name}", val, sync_dist=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.inferer(x, self.model)
        loss, metrics = self.loss_fn(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for name, val in metrics.items():
            self.log(f"val_{name}", val, sync_dist=True)
            
        if batch_idx == 0:
            self._log_images(x, y, y_hat, "val")
            
        return loss

    def _log_images(self, x, y, y_hat, stage):
        # Extract central slice
        d_idx = x.shape[2] // 2
        img = x[0, 0, d_idx].detach().cpu().numpy()
        lbl = y[0, 0, d_idx].detach().cpu().numpy()
        pred = y_hat[0, 0, d_idx].detach().cpu().numpy()
        
        # Create side-by-side plot using Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title("Input (EPI)")
        axes[1].imshow(lbl, cmap='gray'); axes[1].set_title("Target (FLAIR*)")
        axes[2].imshow(pred, cmap='gray'); axes[2].set_title(f"Prediction (Epoch {self.current_epoch})")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        
        # 1. Save to disk with unique epoch name
        epoch_path = f"vis/epoch_{self.current_epoch}_{stage}.png"
        latest_path = f"vis/latest_{stage}_comparison.png"
        plt.savefig(epoch_path)
        plt.savefig(latest_path) # Also keep a 'latest' for easy checking
        
        # 2. Log to MLflow if available
        for logger in self.loggers:
            if hasattr(logger, "log_image") and not hasattr(logger, "experiment"):
                try:
                    logger.log_image(key=f"{stage}_comparison", image=epoch_path)
                except:
                    pass
            elif hasattr(logger, "experiment"):
                if hasattr(logger.experiment, "add_image"):
                    combined = np.concatenate([img, lbl, pred], axis=1)
                    logger.experiment.add_image(f"{stage}_epoch_{self.current_epoch}", combined[np.newaxis, ...], self.global_step)
                elif hasattr(logger.experiment, "log_artifact"):
                    logger.experiment.log_artifact(local_path=epoch_path, artifact_path="val_visualizations")

        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        return optimizer
