import torch.nn as nn
from monai.networks.nets import UNETR


class DeepFLAIRUNETR(nn.Module):
    """MONAI UNETR with ReLU final activation.

    Uses the standard ViT encoder + CNN decoder from Hatamizadeh et al. 2022.
    Designed for 64^3 patches, feature_size=16 matches the UNet base_channels default.
    """

    def __init__(
        self,
        img_size: tuple = (64, 64, 64),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 16,
    ):
        super().__init__()
        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
        self.final_activation = nn.ReLU()

    def forward(self, x):
        return self.final_activation(self.unetr(x))
