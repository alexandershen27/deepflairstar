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
        activation: str = "relu",
        norm_name: str = "instance",
    ):
        super().__init__()

        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            norm_name=norm_name,
            spatial_dims=3,
        )

        if activation == "hardsigmoid":
            self.final_activation = nn.Hardsigmoid()
        elif activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif activation == "none":
            self.final_activation = nn.Identity()
        else:
            self.final_activation = nn.ReLU()

    def forward(self, x):
        x = self.swin(x)
        return self.final_activation(x)