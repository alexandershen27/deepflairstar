import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class DeepFLAIRUNETR(nn.Module):
    """
    UNETR: Transformer-based 3D Segmentation/Synthesis network.
    Uses a ViT encoder and U-Net decoder.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple = (64, 64, 64),
        feature_size: int = 16, # Controls the decoder width
        hidden_size: int = 768, # ViT embedding dimension
        mlp_dim: int = 3072,    # ViT MLP dimension
        num_heads: int = 12,    # ViT heads
        pos_embed: str = "perceptron",
        norm_name: str = "instance",
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            res_block=res_block,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        # UNETR expects img_size to match exactly
        return self.model(x)
