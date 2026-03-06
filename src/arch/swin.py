import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24, # Must be divisible by 12 per MONAI constraint
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        # Ensure feature_size is valid for SwinUNETR
        if feature_size % 12 != 0:
            print(f"--- WARNING: Adjusting Swin feature_size from {feature_size} to 24 for compliance ---")
            feature_size = 24

        self.model = SwinUNETR(
            img_size,
            in_channels,
            out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, x):
        return self.model(x)
