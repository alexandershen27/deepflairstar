import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        dy_pred = y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]
        dy_true = y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]
        
        dx_pred = y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]
        dx_true = y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :]
        
        dz_pred = y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1]
        dz_true = y_true[:, :, :, :, 1:] - y_true[:, :, :, :, :-1]
        
        loss_y = F.l1_loss(dy_pred, dy_true)
        loss_x = F.l1_loss(dx_pred, dx_true)
        loss_z = F.l1_loss(dz_pred, dz_true)
        
        return (loss_x + loss_y + loss_z) / 3.0

class DeepFLAIRLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0, grad_weight=1.0):
        super().__init__()
        # 1. MSE Loss for large-scale error penalization
        self.mse = nn.MSELoss()

        # 2. Restoration: Using standard 1.0 range for stability
        self.ssim = SSIMLoss(spatial_dims=3, data_range=1.0)

        self.grad = GradientLoss()

        self.w_mse = mse_weight
        self.w_ssim = ssim_weight
        self.w_grad = grad_weight

    def forward(self, y_pred, y_true):
        l_mse = self.mse(y_pred, y_true)
        l_ssim = self.ssim(y_pred, y_true)
        l_grad = self.grad(y_pred, y_true)

        total_loss = (self.w_mse * l_mse) + (self.w_ssim * l_ssim) + (self.w_grad * l_grad)

        return total_loss, {"mse": l_mse, "ssim_loss": l_ssim, "grad_loss": l_grad}
