import torch
import numpy as np
import pytorch_lightning as pl
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
        
        # Sliding window for validation/testing
        self.inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        
        loss, metrics = self.loss_fn(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        for name, val in metrics.items():
            self.log(f"train_{name}", val)
            
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        
        # Use sliding window for full-volume validation
        y_hat = self.inferer(x, self.model)
        
        loss, metrics = self.loss_fn(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)
        for name, val in metrics.items():
            self.log(f"val_{name}", val)
            
        if self.logger is not None and batch_idx == 0:
            self._log_images(x, y, y_hat, "val")
            
        return loss

    def _log_images(self, x, y, y_hat, stage):
        # Find the TensorBoard logger if available
        tb_logger = None
        if isinstance(self.logger, list):
            for l in self.logger:
                if hasattr(l, "experiment") and hasattr(l.experiment, "add_image"):
                    tb_logger = l
                    break
        elif hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_image"):
            tb_logger = self.logger

        if tb_logger is None:
            return
            
        # Log central slice to TensorBoard
        # x shape: [B, C, D, H, W]
        d_idx = x.shape[2] // 2
        img = x[0, 0, d_idx].detach().cpu().numpy()
        lbl = y[0, 0, d_idx].detach().cpu().numpy()
        pred = y_hat[0, 0, d_idx].detach().cpu().numpy()
        
        # Stack images horizontally
        combined = np.concatenate([img, lbl, pred], axis=1)
        
        # Add channel dimension for TB [C, H, W]
        combined_tb = combined[np.newaxis, ...]
        
        tb_logger.experiment.add_image(
            f"{stage}_comparison", 
            combined_tb, 
            self.global_step
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        return optimizer
