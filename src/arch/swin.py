import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    """
    Swin UNETR: 3D Transformer-based Synthesis.
    Audited for MONAI 1.3+ compatibility.
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
        
        # We ensure feature_size is exactly 24 or 48 (MONAI gold standards)
        # to guarantee compatibility with internal attention head math.
        safe_feature_size = 24 if feature_size % 12 != 0 else feature_size

        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=safe_feature_size,
            use_checkpoint=use_checkpoint,
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, x):
        # Input tensor shape: [B, 1, 64, 64, 64]
        return self.model(x)
