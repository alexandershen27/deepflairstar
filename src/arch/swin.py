import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    """
    Optimized Swin UNETR for MRI Synthesis.
    Using feature_size=24 for 24GB GPU balance.
    """
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24, 
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        # We wrap the MONAI implementation
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name="instance", # Swapped to instance for synthesis stability
            spatial_dims=3,
        )

    def forward(self, x):
        # Input: [B, C, D, H, W]
        return self.model(x)
