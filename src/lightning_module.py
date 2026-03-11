import torch
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
            # Ensure feature_size is divisible by 12 for MONAI SwinUNETR
            # Default to 24 if the provided base_channels (e.g. 16) is invalid
            swin_features = base_channels if base_channels % 12 == 0 else 24
            self.model = DeepFLAIRSwin(feature_size=swin_features, img_size=patch_size)
        else:
            self.model = DeepFLAIRNet(base_channels=base_channels)
            
        self.loss_fn = DeepFLAIRLoss(
            mse_weight=mse_weight,
            ssim_weight=ssim_weight, 
            grad_weight=grad_weight
        )
        
        # Register 3D Hann Window as a buffer so it moves to GPU automatically
        self.register_buffer("hann_window", self._generate_3d_hann(patch_size))

    def _generate_3d_hann(self, patch_size):
        w_d = torch.hann_window(patch_size[0], periodic=False)
        w_h = torch.hann_window(patch_size[1], periodic=False)
        w_w = torch.hann_window(patch_size[2], periodic=False)
        window_3d = w_d.view(-1, 1, 1) * w_h.view(1, -1, 1) * w_w.view(1, 1, -1)
        return window_3d.unsqueeze(0)

    def forward(self, x):
        return self.model(x)

    def _calculate_psnr(self, mse_loss):
        if mse_loss == 0:
            return 100.0
        return 20 * torch.log10(1.0 / torch.sqrt(mse_loss))

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.model(x)
        loss, metrics = self.loss_fn(y_hat, y)
        bs = x.shape[0]
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("train_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        self.log("train_ssim", metrics["ssim_loss"], sync_dist=True, batch_size=bs)
        self.log("train_grad", metrics["grad_loss"], sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
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
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        self.log("val_ssim", metrics["ssim_loss"], sync_dist=True, batch_size=bs)
        self.log("val_grad", metrics["grad_loss"], sync_dist=True, batch_size=bs)
        
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
        psnr = self._calculate_psnr(metrics["mse"])
        
        self.log("test_loss", loss, sync_dist=True, batch_size=bs)
        self.log("test_mse", metrics["mse"], sync_dist=True, batch_size=bs)
        self.log("test_psnr", psnr, sync_dist=True, batch_size=bs)
        self.log("test_ssim", metrics["ssim_loss"], sync_dist=True, batch_size=bs)
        self.log("test_grad", metrics["grad_loss"], sync_dist=True, batch_size=bs)
        
        self._save_test_result(batch, y_hat)
        
        if batch_idx == 0:
            self._log_images(x, y, y_hat, "test")
        return loss

    def _save_test_result(self, batch, y_hat):
        save_dir = os.path.join("outputs", "test_results")
        if self.logger and hasattr(self.logger, "name"):
            save_dir = os.path.join("outputs", self.logger.name, "test_outputs")
        os.makedirs(save_dir, exist_ok=True)
        
        subj_id = batch.get("subject_id", ["unknown"])[0]
        meta_dict = batch.get("image_meta_dict", {})
        
        orig_shape = meta_dict.get("spatial_shape", None)
        output_vol = y_hat[0, 0].detach().cpu().numpy()
        
        if orig_shape is not None:
            s = orig_shape[0].tolist() if torch.is_tensor(orig_shape) else orig_shape
            p = output_vol.shape
            starts = [(pi - si) // 2 for pi, si in zip(p, s)]
            output_vol = output_vol[
                starts[0] : starts[0] + s[0],
                starts[1] : starts[1] + s[1],
                starts[2] : starts[2] + s[2]
            ]
        
        affine = meta_dict.get("affine", None)
        if affine is not None:
            affine = affine[0].cpu().numpy() if torch.is_tensor(affine) else affine
        else:
            affine = np.eye(4)
            
        new_img = nib.Nifti1Image(output_vol, affine)
        save_path = os.path.join(save_dir, f"{subj_id}_synthetic_flair.nii.gz")
        nib.save(new_img, save_path)

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
