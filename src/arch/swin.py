import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24, 
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        # Using positional arguments for img_size, in_channels, out_channels
        # This is the most stable way across MONAI 1.0 -> 1.3
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
