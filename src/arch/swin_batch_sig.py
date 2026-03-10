import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class DeepFLAIRSwinBatchSig(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_size=24, img_size=(64,64,64)):
        super().__init__()
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            norm_name="batch",
            spatial_dims=3
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        return self.final_activation(self.swin(x))
