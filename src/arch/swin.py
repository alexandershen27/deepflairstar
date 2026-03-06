import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    """
    Swin UNETR: Hierarchical Transformer for 3D Synthesis.
    More memory efficient and captures multiscale features better than standard UNETR.
    """
    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24, # Controls the width of the hierarchical stages
        use_checkpoint: bool = False, # Enable if VRAM is tight
    ):
        super().__init__()
        
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        # Swin UNETR expects patch sizes that are multiples of 32
        return self.model(x)
