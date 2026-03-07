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
        
        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name="batch", # Per user requirement
            spatial_dims=3,
        )
        
        # HardSigmoid: Aggressive, linear $[0, 1]$ clamping.
        self.final_activation = nn.Hardsigmoid()

    def forward(self, x):
        x = self.swin(x)
        return self.final_activation(x)
