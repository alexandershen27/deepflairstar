import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwin(nn.Module):
    """
    Swin UNETR: Dynamic-size implementation.
    Audited against server-specific MONAI signature.
    """
    def __init__(
        self,
        img_size: tuple = (64, 64, 64), # Argument kept for class parity but unused in call
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24, 
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        # Enforce divisible by 12 per MONAI requirement
        safe_feature_size = feature_size if feature_size % 12 == 0 else 24

        # Based on titan2 signature: 
        # (self, in_channels, out_channels, patch_size=2, depths=(2,2,2,2), ...)
        # Note: img_size is NOT part of the signature!
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=safe_feature_size,
            use_checkpoint=use_checkpoint,
            norm_name="instance",
            spatial_dims=3,
        )

    def forward(self, x):
        return self.model(x)
